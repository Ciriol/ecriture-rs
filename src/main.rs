//! UI egui : canvas d'écriture + bouton de reconnaissance.

mod ink;
mod recognizer;

use eframe::egui;
use ink::{Point, Stroke};
use recognizer::{CandleRecognizer, MockRecognizer, Recognizer};

/// Délai de pause (en secondes) après lequel on reconnaît les nouveaux strokes.
const DEBOUNCE_SECS: f64 = 0.6;

struct InkApp {
    strokes: Vec<Stroke>,
    current: Option<Stroke>,
    result: String,
    /// Nombre de strokes déjà intégrés dans `result`.
    recognized_up_to: usize,
    /// Temps (s) du dernier point tracé.
    last_draw_time: f64,
    recognizer: Box<dyn Recognizer>,
}

impl InkApp {
    fn new(recognizer: Box<dyn Recognizer>) -> Self {
        Self {
            strokes: Vec::new(),
            current: None,
            result: String::new(),
            recognized_up_to: 0,
            last_draw_time: 0.0,
            recognizer,
        }
    }

    /// Reconnaissance complète (bouton) : on repart de zéro.
    fn recognize_all(&mut self) {
        if self.strokes.is_empty() && self.current.is_none() {
            self.result = "( rien à reconnaître )".to_string();
            return;
        }
        let img = ink::render_to_image(&self.strokes, 64);
        match self.recognizer.recognize(&img) {
            Ok(text) => self.result = text,
            Err(e) => self.result = format!("erreur: {e:#}"),
        }
        self.recognized_up_to = self.strokes.len();
    }

    /// Reconnaissance incrémentale : uniquement les strokes tracés
    /// depuis le dernier appel, ajoutés à la suite du texte reconnu.
    fn recognize_pending(&mut self) {
        let pending = &self.strokes[self.recognized_up_to..];
        if pending.is_empty() {
            return;
        }
        let img = ink::render_to_image(pending, 64);
        match self.recognizer.recognize(&img) {
            Ok(text) => {
                if !self.result.is_empty() && !text.is_empty() {
                    self.result.push(' ');
                }
                self.result.push_str(&text);
            }
            Err(e) => self.result = format!("erreur: {e:#}"),
        }
        self.recognized_up_to = self.strokes.len();
    }
}

impl eframe::App for InkApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("ecriture-rs");
                if ui.button("Reconnaître").clicked() {
                    self.recognize_all();
                }
                if ui.button("Effacer").clicked() {
                    self.strokes.clear();
                    self.current = None;
                    self.result.clear();
                    self.recognized_up_to = 0;
                }
                ui.label(format!("{} strokes", self.strokes.len()));
            });
        });

        egui::Panel::bottom("result").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Texte reconnu :");
                ui.label(egui::RichText::new(&self.result).monospace());
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::drag());

            // Fond blanc du papier.
            painter.rect_filled(response.rect, 0.0, egui::Color32::WHITE);

            // Rendu des strokes terminés.
            for stroke in &self.strokes {
                draw_stroke(&painter, stroke, response.rect);
            }
            if let Some(stroke) = &self.current {
                draw_stroke(&painter, stroke, response.rect);
            }

            // Capture du tracé.
            let now = ui.input(|i| i.time);
            if response.drag_started() {
                self.current = Some(Stroke::new());
            }
            if response.dragged() {
                if let Some(stroke) = &mut self.current {
                    let pos = response.hover_pos().unwrap_or(response.rect.min);
                    stroke.push(Point {
                        x: pos.x - response.rect.min.x,
                        y: pos.y - response.rect.min.y,
                        t: now,
                        pressure: 1.0,
                    });
                    self.last_draw_time = now;
                }
            }
            if response.drag_stopped() {
                if let Some(stroke) = self.current.take() {
                    if !stroke.is_empty() {
                        self.strokes.push(stroke);
                        self.last_draw_time = now;
                    }
                }
            }
        });

        // Reconnaissance incrémentale : après une pause de tracé,
        // on reconnaît uniquement les nouveaux strokes et on les ajoute au texte.
        let now = ui.ctx().input(|i| i.time);
        let pending = self.strokes.len().saturating_sub(self.recognized_up_to);
        if pending > 0 && now - self.last_draw_time > DEBOUNCE_SECS {
            self.recognize_pending();
        } else if pending > 0 {
            // On reprogramme un rafraîchissement pour surveiller la pause.
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

fn draw_stroke(painter: &egui::Painter, stroke: &Stroke, rect: egui::Rect) {
    if stroke.points.len() < 2 {
        return;
    }
    let points: Vec<egui::Pos2> = stroke
        .points
        .iter()
        .map(|p| egui::pos2(rect.min.x + p.x, rect.min.y + p.y))
        .collect();
    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(2.5, egui::Color32::BLACK),
    ));
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "ecriture-rs",
        options,
        Box::new(|_cc| {
            // Modèle réel si présent, sinon bouchon pour tester l'UI.
            let recognizer: Box<dyn Recognizer> =
                match CandleRecognizer::new("model.onnx", "chars.txt", 64) {
                    Ok(r) => Box::new(r),
                    Err(_) => {
                        eprintln!("model.onnx absent : utilisation du MockRecognizer");
                        Box::new(MockRecognizer)
                    }
                };
            Ok(Box::new(InkApp::new(recognizer)))
        }),
    )
}
