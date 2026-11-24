# ecriture-rs : squelette de départ

Application Rust de prise de notes avec reconnaissance d'écriture quasi temps réel.
Pipeline : capture de strokes (egui) → rendu en image → inférence ONNX locale (candle) → texte.

Arborescence du projet :

```text
ecriture-rs/
├── Cargo.toml
├── README.md
└── src/
    ├── main.rs         (UI egui + boucle d'événements)
    ├── ink.rs          (strokes, prétraitement, rendu image)
    └── recognizer.rs   (trait Recognizer + impl candle-onnx + mock)
```

## Cargo.toml

```toml
[package]
name = "ecriture-rs"
version = "0.1.0"
edition = "2024"

[dependencies]
eframe = "0.37"
egui = "0.37"
candle-core = "0.11"
candle-onnx = "0.11"
image = "0.25"
anyhow = "1.0"
```

## src/ink.rs

```rust
//! Capture et prétraitement des strokes (encre numérique).

use image::{GrayImage, Luma};

/// Un point d'encre : position, temps (ms) et pression (0.0..1.0).
#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub t: f64,
    pub pressure: f32,
}

/// Un stroke = une séquence de points (un lever de stylet).
#[derive(Clone, Debug, Default)]
pub struct Stroke {
    pub points: Vec<Point>,
}

impl Stroke {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    pub fn push(&mut self, p: Point) {
        // Évite les points dupliqués (bruits de capteur).
        if let Some(last) = self.points.last() {
            let dx = p.x - last.x;
            let dy = p.y - last.y;
            if (dx * dx + dy * dy) < 0.25 {
                return;
            }
        }
        self.points.push(p);
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Rééchantillonnage à pas constant : utile pour les modèles
    /// entraînés sur des séquences normalisées.
    pub fn resample(&self, step: f32) -> Vec<(f32, f32)> {
        if self.points.len() < 2 {
            return self.points.iter().map(|p| (p.x, p.y)).collect();
        }
        let mut out = Vec::new();
        let pts: Vec<(f32, f32)> = self.points.iter().map(|p| (p.x, p.y)).collect();
        out.push(pts[0]);
        let mut acc = 0.0;
        for w in pts.windows(2) {
            let (x0, y0) = w[0];
            let (x1, y1) = w[1];
            let d = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
            let mut s = step - acc;
            while s <= d {
                let r = if d > 0.0 { s / d } else { 0.0 };
                out.push((x0 + r * (x1 - x0), y0 + r * (y1 - y0)));
                s += step;
            }
            acc = d - (s - step);
        }
        out
    }
}

/// Boîte englobante de tous les strokes.
pub fn bounding_box(strokes: &[Stroke]) -> Option<(f32, f32, f32, f32)> {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for s in strokes {
        for p in &s.points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
    }
    if min_x > max_x {
        None
    } else {
        Some((min_x, min_y, max_x, max_y))
    }
}

/// Rend les strokes dans une image grayscale normalisée (hauteur fixe,
/// largeur proportionnelle) : c'est le format d'entrée des modèles OCR.
pub fn render_to_image(strokes: &[Stroke], height: u32) -> GrayImage {
    let (min_x, min_y, max_x, max_y) = bounding_box(strokes).unwrap_or((0.0, 0.0, 1.0, 1.0));
    let w = (max_x - min_x).max(1.0);
    let h = (max_y - min_y).max(1.0);
    let scale = height as f32 / h;
    let width = ((w * scale) as u32).max(1);

    let mut img = GrayImage::from_pixel(width, height, Luma([255]));
    let thickness = (height as f32 / 32.0).max(1.5);

    for stroke in strokes {
        let pts = stroke.resample(thickness / 2.0);
        for w in pts.windows(2) {
            let (x0, y0) = w[0];
            let (x1, y1) = w[1];
            draw_line(
                &mut img,
                (x0 - min_x) * scale,
                (y0 - min_y) * scale,
                (x1 - min_x) * scale,
                (y1 - min_y) * scale,
                thickness,
            );
        }
    }
    img
}

/// Tracé de segment épais (disques le long de la ligne).
fn draw_line(img: &mut GrayImage, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32) {
    let d = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
    let steps = (d / (thickness / 4.0)).ceil().max(1.0) as u32;
    let r = thickness / 2.0;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let cx = x0 + t * (x1 - x0);
        let cy = y0 + t * (y1 - y0);
        for dy in -(r as i32)..=(r as i32) {
            for dx in -(r as i32)..=(r as i32) {
                if (dx * dx + dy * dy) as f32 <= r * r {
                    let px = cx as i32 + dx;
                    let py = cy as i32 + dy;
                    if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                        img.put_pixel(px as u32, py as u32, Luma([0]));
                    }
                }
            }
        }
    }
}
```

## src/recognizer.rs

```rust
//! Inférence de reconnaissance d'écriture via ONNX (candle), 100 % locale.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use image::GrayImage;

/// Alphabet du modèle. Le token CTC "blank" est l'index 0.
/// Fichier chars.txt : un caractère par ligne, dans l'ordre du modèle.
pub fn load_charset(path: &str) -> Result<Vec<char>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("charset introuvable: {}", path))?;
    let mut chars: Vec<char> = content.chars().filter(|c| *c != '\n').collect();
    // On réserve l'index 0 pour le blank CTC.
    chars.insert(0, '\u{0}');
    Ok(chars)
}

pub trait Recognizer {
    fn recognize(&mut self, img: &GrayImage) -> Result<String>;
}

/// Décodage glouton CTC : argmax par pas de temps, dédoublonnage,
/// suppression du blank (index 0).
pub fn ctc_greedy_decode(logits: &[f32], width: usize, height: usize, charset: &[char]) -> String {
    let mut out = String::new();
    let mut prev = 0usize;
    for t in 0..width {
        let mut best = 0usize;
        let mut best_v = f32::MIN;
        for c in 0..height {
            let v = logits[t * height + c];
            if v > best_v {
                best_v = v;
                best = c;
            }
        }
        if best != 0 && best != prev {
            if let Some(&ch) = charset.get(best) {
                out.push(ch);
            }
        }
        prev = best;
    }
    out
}

/// Reconnaissance via un modèle ONNX simple (CRNN / CNN-BiLSTM-CTC à graphe unique).
/// Entrée : image grayscale normalisée [1, 1, H, W] en f32 (0..1, fond blanc = 1).
/// Sortie : logits [1, T, num_classes].
pub struct CandleRecognizer {
    model: candle_onnx::Model,
    charset: Vec<char>,
    device: Device,
    input_h: u32,
}

impl CandleRecognizer {
    pub fn new(onnx_path: &str, charset_path: &str, input_h: u32) -> Result<Self> {
        let model = candle_onnx::Model::load(onnx_path)
            .with_context(|| format!("modèle ONNX introuvable: {}", onnx_path))?;
        let charset = load_charset(charset_path)?;
        Ok(Self {
            model,
            charset,
            device: Device::Cpu,
            input_h,
        })
    }

    fn image_to_tensor(&self, img: &GrayImage) -> Result<Tensor> {
        // Redimensionne à hauteur fixe en gardant les proportions.
        let ratio = self.input_h as f32 / img.height() as f32;
        let w = ((img.width() as f32 * ratio) as u32).max(1);
        let resized = image::imageops::resize(
            img,
            w,
            self.input_h,
            image::imageops::FilterType::Lanczos3,
        );
        let mut data = vec![0f32; (w * self.input_h) as usize];
        for (i, p) in resized.pixels().enumerate() {
            // 255 (blanc) -> 1.0, 0 (noir) -> 0.0
            data[i] = p.0[0] as f32 / 255.0;
        }
        Tensor::from_vec(data, (1, 1, self.input_h as usize, w as usize), &self.device)
            .context("construction du tenseur d'entrée")
    }
}

impl Recognizer for CandleRecognizer {
    fn recognize(&mut self, img: &GrayImage) -> Result<String> {
        let input = self.image_to_tensor(img)?;
        let outputs = self.model.forward(&[&input]).context("inférence ONNX")?;
        let logits = outputs[0]
            .to_device(&Device::Cpu)
            .context("lecture des logits")?;
        let shape = logits.dims();
        // shape attendu : [1, T, C]
        let (t, c) = (shape[1], shape[2]);
        let flat = logits
            .flatten_all()
            .context("aplatissement des logits")?
            .to_vec1::<f32>()
            .context("conversion f32")?;
        Ok(ctc_greedy_decode(&flat, t, c, &self.charset))
    }
}

/// Bouchon pour faire tourner l'UI sans modèle (renvoie une chaîne factice).
pub struct MockRecognizer;

impl Recognizer for MockRecognizer {
    fn recognize(&mut self, _img: &GrayImage) -> Result<String> {
        Ok("[reconnaissance simulée]".to_string())
    }
}
```

## src/main.rs

```rust
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
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("bar").show(ctx, |ui| {
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

        egui::BottomPanel::bottom("result").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Texte reconnu :");
                ui.label(egui::RichText::new(&self.result).monospace());
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
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
            if response.drag_released() {
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
        let now = ctx.input(|i| i.time);
        let pending = self.strokes.len().saturating_sub(self.recognized_up_to);
        if pending > 0 && now - self.last_draw_time > DEBOUNCE_SECS {
            self.recognize_pending();
        } else if pending > 0 {
            // On reprogramme un rafraîchissement pour surveiller la pause.
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
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
```

## README.md

````markdown
# ecriture-rs

Prise de notes avec reconnaissance d'écriture locale (Rust, egui, candle/ONNX).

## Prérequis

1. Rust (rustup).
2. Un modèle ONNX de reconnaissance d'écriture + son alphabet :
   - Option simple : exporter un CRNN (CNN-BiLSTM-CTC) entraîné sur IAM en ONNX.
   - Option TrOCR (Microsoft, licence MIT) : export via `optimum` (encoder/decoder séparés,
     décodage glouton côté Rust à adapter dans `recognizer.rs`).
3. Placer `model.onnx` et `chars.txt` (un caractère par ligne, ordre = classes du modèle)
   à la racine du projet.

## Lancer

```bash
cargo run --release
```

Sans modèle, l'app démarre avec un reconnaisseur simulé (pour valider l'UI).

## Prochaines étapes

- Modèle stroke-based (séquences x, y, t) au lieu du rendu image, pour un vrai temps réel.
- Fenêtre glissante avec hypothèses révisables (le texte se corrige en direct).
- Segmentation en mots/gestes (double-tap, retour à la ligne).
````
