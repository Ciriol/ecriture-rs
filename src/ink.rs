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
