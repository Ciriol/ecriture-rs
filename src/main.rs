//! UI egui : canvas d'écriture + bouton de reconnaissance.

mod ink;
mod recognizer;
mod tablet;

use std::sync::mpsc::{channel, Receiver};

use eframe::egui;
use ink::{Point, Stroke};
use recognizer::{CandleRecognizer, MockRecognizer, Recognizer};
use tablet::TabletEvent;

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
    /// Un touch (stylet/doigt) est actuellement actif.
    touch_active: bool,
    /// Dernière position connue d'un touch (si le backend n'émule pas la souris).
    last_touch_pos: Option<egui::Pos2>,
    /// Mode debug : affiche les événements d'entrée reçus.
    debug: bool,
    debug_info: String,
    /// Réception des événements du stylet lus via evdev.
    tablet_rx: Option<Receiver<TabletEvent>>,
    /// Compteur d'événements tablette reçus (pour le mode debug).
    tablet_count: u64,
}

impl InkApp {
    fn new(recognizer: Box<dyn Recognizer>) -> Self {
        // Lance la lecture evdev des périphériques stylet/touch en arrière-plan.
        let (tx, rx) = channel();
        tablet::spawn_readers(tx);
        Self {
            strokes: Vec::new(),
            current: None,
            result: String::new(),
            recognized_up_to: 0,
            last_draw_time: 0.0,
            recognizer,
            touch_active: false,
            last_touch_pos: None,
            debug: false,
            debug_info: String::new(),
            tablet_rx: Some(rx),
            tablet_count: 0,
        }
    }

    /// Ajoute un point au stroke en cours (coordonnées relatives au canvas).
    fn push_point(&mut self, p: egui::Pos2, pressure: f32, now: f64, rect: egui::Rect) {
        if let Some(stroke) = &mut self.current {
            stroke.push(Point {
                x: p.x - rect.min.x,
                y: p.y - rect.min.y,
                t: now,
                pressure,
            });
            self.last_draw_time = now;
        }
    }

    /// Termine le stroke en cours et l'archive.
    fn end_stroke(&mut self, now: f64) {
        if let Some(stroke) = self.current.take() {
            if !stroke.is_empty() {
                self.strokes.push(stroke);
                self.last_draw_time = now;
            }
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
                ui.checkbox(&mut self.debug, "debug");
            });
        });

        egui::Panel::bottom("result").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Texte reconnu :");
                ui.label(egui::RichText::new(&self.result).monospace());
            });
            if self.debug {
                ui.label(egui::RichText::new(&self.debug_info).monospace().small());
            }
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

            // Capture du tracé. On ne se fie pas qu'à la Response du canvas :
            // on lit l'état global du pointer (souris, ou stylet émulé en
            // bouton principal) ET les événements touch bruts — un stylet
            // sous Wayland arrive en Event::Touch avec une force (pression).
            let now = ui.input(|i| i.time);
            let rect = response.rect;

            let (mut touch_started, mut touch_ended, mut pressure, mut touch_pos) =
                (false, false, 1.0f32, None);
            ui.input(|i| {
                for ev in &i.events {
                    if let egui::Event::Touch { phase, pos, force, .. } = ev {
                        match phase {
                            egui::TouchPhase::Start => {
                                touch_started = true;
                                touch_pos = Some(*pos);
                            }
                            egui::TouchPhase::Move => {
                                touch_pos = Some(*pos);
                            }
                            egui::TouchPhase::End | egui::TouchPhase::Cancel => {
                                touch_ended = true;
                            }
                        }
                        if let Some(f) = force {
                            pressure = *f;
                        }
                    }
                }
            });
            if touch_started {
                self.touch_active = true;
            }
            if let Some(p) = touch_pos {
                self.last_touch_pos = Some(p);
            }
            if touch_ended {
                self.touch_active = false;
                self.last_touch_pos = None;
            }

            let (pressed, down, released, pointer_pos) = ui.input(|i| {
                let p = &i.pointer;
                (p.any_pressed(), p.any_down(), p.any_released(), p.latest_pos())
            });
            let pos = pointer_pos.or(self.last_touch_pos);
            let pos_in = pos.filter(|p| rect.contains(*p));
            let drawing = down || self.touch_active;

            if (pressed || touch_started) && pos_in.is_some() {
                self.current = Some(Stroke::new());
            }
            if drawing && !touch_ended {
                if let Some(p) = pos {
                    if rect.contains(p) {
                        self.push_point(p, pressure, now, rect);
                    }
                }
            }
            if released || touch_ended {
                self.end_stroke(now);
            }

            // Tracé au stylet lu directement via evdev (winit ne remonte pas
            // toujours le périphérique à egui). Coordonnées normalisées 0..1
            // de l'écran -> points UI via le screen_rect.
            let mut tablet_events = Vec::new();
            if let Some(rx) = &self.tablet_rx {
                while let Ok(ev) = rx.try_recv() {
                    tablet_events.push(ev);
                }
            }
            if !tablet_events.is_empty() {
                let screen = ui.ctx().viewport_rect();
                for ev in tablet_events {
                    self.tablet_count += 1;
                    match ev {
                        TabletEvent::Down { x, y, pressure } => {
                            let p = egui::pos2(x * screen.width(), y * screen.height());
                            if rect.contains(p) {
                                self.current = Some(Stroke::new());
                                self.push_point(p, pressure, now, rect);
                            }
                        }
                        TabletEvent::Move { x, y, pressure } => {
                            let p = egui::pos2(x * screen.width(), y * screen.height());
                            if rect.contains(p) {
                                self.push_point(p, pressure, now, rect);
                            }
                        }
                        TabletEvent::Up => {
                            self.end_stroke(now);
                        }
                    }
                }
            }

            // Mode debug : remonte ce que egui reçoit comme entrée.
            if self.debug {
                self.debug_info = ui.input(|i| {
                    let evs: Vec<String> = i
                        .events
                        .iter()
                        .filter_map(|ev| match ev {
                            egui::Event::PointerButton { button, pressed, pos, .. } => Some(
                                format!(
                                    "button {:?} {} @ {:.0},{:.0}",
                                    button,
                                    if *pressed { "down" } else { "up" },
                                    pos.x,
                                    pos.y
                                ),
                            ),
                            egui::Event::Touch { phase, pos, force, .. } => Some(format!(
                                "touch {:?} @ {:.0},{:.0} force={:?}",
                                phase, pos.x, pos.y, force
                            )),
                            _ => None,
                        })
                        .collect();
                    format!(
                        "down={} pressed={} released={} any_touches={} pos={:?}\n{}",
                        i.pointer.any_down(),
                        i.pointer.any_pressed(),
                        i.pointer.any_released(),
                        i.any_touches(),
                        i.pointer.latest_pos(),
                        evs.join("\n")
                    )
                });
                self.debug_info
                    .push_str(&format!("\ntablet events: {}", self.tablet_count));
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
