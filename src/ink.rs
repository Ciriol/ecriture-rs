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

    /// Rééchantillonne à exactement `n` points, par longueur d'arc,
    /// en interpolant aussi le temps et la pression. Pour les modèles
    /// de trajectoires (entrée [1, T, F]).
    pub fn resample_n_points(&self, n: usize) -> Vec<Point> {
        if self.points.len() < 2 || n < 2 {
            return self.points.clone();
        }
        let mut len = 0.0f32;
        for w in self.points.windows(2) {
            len += ((w[1].x - w[0].x).powi(2) + (w[1].y - w[0].y).powi(2)).sqrt();
        }
        if len <= 0.0 {
            return vec![self.points[0]; n];
        }
        let step = len / (n - 1) as f32;
        let mut out = Vec::with_capacity(n);
        out.push(self.points[0]);
        let mut acc = 0.0f32;
        for w in self.points.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let d = ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt();
            let mut s = step - acc;
            while s <= d {
                let r = if d > 0.0 { s / d } else { 0.0 };
                out.push(Point {
                    x: a.x + r * (b.x - a.x),
                    y: a.y + r * (b.y - a.y),
                    t: a.t + r as f64 * (b.t - a.t),
                    pressure: a.pressure + r * (b.pressure - a.pressure),
                });
                s += step;
            }
            acc = d - (s - step);
        }
        while out.len() < n {
            out.push(*out.last().unwrap());
        }
        out.truncate(n);
        out
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

/// Rend les strokes dans une image grayscale (hauteur fixe `height`) :
/// c'est le format d'entrée des modèles OCR de lignes de texte.
///
/// Le modèle CRNN-IAM attend une image de *ligne de texte* avec des marges
/// (comme les crops IAM d'origine) : on centre la bounding box des strokes
/// dans l'image en ne lui donnant que ~75 % de la hauteur, avec un padding
/// horizontal d'au moins `height` de chaque côté et une largeur minimale de
/// `2 * height` (évite les images trop étroites, pauvres en pas de temps CTC).
/// L'épaisseur du trait est proportionnelle à la hauteur (~height/10), proche
/// de l'encre IAM à 32 px (~3 px une fois redimensionnée).
pub fn render_to_image(strokes: &[Stroke], height: u32) -> GrayImage {
    let (min_x, min_y, max_x, max_y) = bounding_box(strokes).unwrap_or((0.0, 0.0, 1.0, 1.0));
    let w = (max_x - min_x).max(1.0);
    let h = (max_y - min_y).max(1.0);
    // La ligne de texte occupe 75 % de la hauteur, centrée verticalement.
    let scale = (height as f32 * 0.75) / h;
    let text_w = w * scale;
    let pad_x = (height as f32 * 0.5).max(4.0);
    let width = ((text_w + 2.0 * pad_x).ceil() as u32).max(height * 2).max(1);
    // Centrage horizontal de la bounding box dans l'image.
    let off_x = (width as f32 - text_w) / 2.0;
    let off_y = (height as f32 - h * scale) / 2.0;

    let mut img = GrayImage::from_pixel(width, height, Luma([255]));
    let thickness = (height as f32 / 10.0).max(2.0);

    for stroke in strokes {
        // Lissage (moyenne glissante) du chemin rééchantillonné : réduit le
        // bruit du capteur (micro-zigzags) qui tache le rendu final.
        let pts = smooth_path(&stroke.resample(thickness / 2.0), 5);
        for w in pts.windows(2) {
            let (x0, y0) = w[0];
            let (x1, y1) = w[1];
            draw_line(
                &mut img,
                off_x + (x0 - min_x) * scale,
                off_y + (y0 - min_y) * scale,
                off_x + (x1 - min_x) * scale,
                off_y + (y1 - min_y) * scale,
                thickness,
            );
        }
    }
    img
}

/// Lissage d'un chemin (x, y) par moyenne glissante de `w` points.
fn smooth_path(pts: &[(f32, f32)], w: usize) -> Vec<(f32, f32)> {
    if pts.len() < 3 || w < 3 {
        return pts.to_vec();
    }
    let half = w / 2;
    (0..pts.len())
        .map(|i| {
            let a = i.saturating_sub(half);
            let b = (i + half).min(pts.len() - 1);
            let n = (b - a + 1) as f32;
            let sx: f32 = pts[a..=b].iter().map(|p| p.0).sum();
            let sy: f32 = pts[a..=b].iter().map(|p| p.1).sum();
            (sx / n, sy / n)
        })
        .collect()
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

/// Nombre de features par point de trajectoire :
/// x, y, dx, dy, dt, pression, cos(direction), sin(direction).
pub const FEATURE_F: usize = 8;

/// Features de trajectoire normalisées pour un groupe de strokes, prêtes à
/// être sérialisées en tenseur [1, T_total, FEATURE_F] pour un modèle de
/// séquences (LRU/transformer). Chaque stroke est rééchantillonné à
/// `t_per_stroke` points ; positions centrées/réduites via la bounding box,
/// dt normalisé (÷ 100 ms, clampé à 1).
pub fn strokes_to_features(strokes: &[Stroke], t_per_stroke: usize) -> Vec<f32> {
    let (min_x, min_y, max_x, max_y) = bounding_box(strokes).unwrap_or((0.0, 0.0, 1.0, 1.0));
    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;
    let scale = (max_x - min_x).max(max_y - min_y).max(1.0);
    let mut out = Vec::with_capacity(strokes.len() * t_per_stroke * FEATURE_F);
    let mut prev: Option<Point> = None;
    for stroke in strokes {
        for p in stroke.resample_n_points(t_per_stroke) {
            let px = (p.x - cx) / scale;
            let py = (p.y - cy) / scale;
            let (dx, dy, dt) = match &prev {
                Some(q) => (
                    (p.x - q.x) / scale,
                    (p.y - q.y) / scale,
                    (((p.t - q.t).max(0.0) / 0.1).min(1.0)) as f32,
                ),
                None => (0.0, 0.0, 0.0),
            };
            let dir = dy.atan2(dx);
            out.extend_from_slice(&[px, py, dx, dy, dt, p.pressure, dir.cos(), dir.sin()]);
            prev = Some(p);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_shape_et_normalisation() {
        let mut s = Stroke::new();
        for i in 0..50 {
            s.push(Point {
                x: i as f32,
                y: (i as f32 / 5.0).sin() * 10.0,
                t: i as f64 * 0.016,
                pressure: 0.5,
            });
        }
        let feats = strokes_to_features(&[s], 32);
        assert_eq!(feats.len(), 32 * FEATURE_F);
        // positions normalisées dans [-1, 1]
        for v in feats.chunks(FEATURE_F) {
            assert!(v[0].abs() <= 1.0 && v[1].abs() <= 1.0);
        }
    }

    #[test]
    fn resample_n_points_compte_exact() {
        let mut s = Stroke::new();
        for i in 0..10 {
            s.push(Point { x: i as f32 * 3.0, y: 0.0, t: i as f64, pressure: 1.0 });
        }
        assert_eq!(s.resample_n_points(16).len(), 16);
        assert_eq!(s.resample_n_points(2).len(), 2);
    }

    #[test]
    fn render_to_image_hauteur_exacte_et_marges() {
        // Une sinusoïde de 200 px de large et 60 px de haut : la bounding box
        // a une vraie hauteur, le trait passe par le milieu vertical.
        let mut s = Stroke::new();
        for i in 0..100 {
            let x = i as f32 * 2.0;
            let y = 30.0 + (i as f32 / 10.0).sin() * 30.0;
            s.push(Point { x, y, t: i as f64 * 0.016, pressure: 1.0 });
        }
        let img = render_to_image(&[s], 64);
        assert_eq!(img.height(), 64);
        // Largeur minimale 2 * height même pour un texte étroit.
        assert!(img.width() >= 2 * 64);
        // Marges : la première et la dernière ligne de l'image restent blanches.
        let top = img.pixels().take(img.width() as usize).all(|p| p.0[0] == 255);
        let bottom_white = img
            .pixels()
            .skip((img.height() as usize - 1) * img.width() as usize)
            .all(|p| p.0[0] == 255);
        assert!(top && bottom_white, "les marges haut/bas doivent être blanches");
        // Le trait est bien présent au milieu de l'image.
        let mid_row_has_ink = img
            .pixels()
            .skip((img.height() as usize / 2) * img.width() as usize)
            .take(img.width() as usize)
            .any(|p| p.0[0] < 128);
        assert!(mid_row_has_ink);
    }

    #[test]
    fn smooth_path_reduit_le_bruit() {
        // zigzag autour de y=0
        let pts: Vec<(f32, f32)> = (0..50)
            .map(|i| (i as f32, if i % 2 == 0 { 5.0 } else { -5.0 }))
            .collect();
        let sm = smooth_path(&pts, 5);
        assert_eq!(sm.len(), pts.len());
        // amplitude lissée < amplitude brute
        let amp = |v: &[(f32, f32)]| {
            let max = v.iter().map(|p| p.1).fold(f32::MIN, f32::max);
            let min = v.iter().map(|p| p.1).fold(f32::MAX, f32::min);
            max - min
        };
        assert!(amp(&sm) < amp(&pts));
        // cas dégénéré
        assert_eq!(smooth_path(&pts[..2], 5).len(), 2);
    }

    #[test]
    fn render_to_image_texte_etroit_largeur_min() {
        // Un seul point -> bounding box 1x1 -> largeur min = 2 * height.
        let mut s = Stroke::new();
        s.push(Point { x: 10.0, y: 10.0, t: 0.0, pressure: 1.0 });
        s.push(Point { x: 10.5, y: 10.0, t: 0.016, pressure: 1.0 });
        let img = render_to_image(&[s], 32);
        assert_eq!(img.height(), 32);
        assert!(img.width() >= 64);
    }

    /// Demo visuelle (à lancer explicitement) : écrit "IN" en bâtons,
    /// sauve le rendu 64 px dans /tmp/strokes_render.png.
    /// `cargo test -- --ignored --nocapture dump_rendu_strokes`
    #[test]
    #[ignore]
    fn dump_rendu_strokes() {
        fn line(x0: f32, y0: f32, x1: f32, y1: f32, t0: f64) -> Stroke {
            let mut s = Stroke::new();
            let n = 40;
            for i in 0..=n {
                let r = i as f32 / n as f32;
                s.push(Point {
                    x: x0 + r * (x1 - x0),
                    y: y0 + r * (y1 - y0),
                    t: t0 + r as f64 * 0.5,
                    pressure: 0.8,
                });
            }
            s
        }
        // "I" puis "N" en bâtons, coordonnées canvas ~ 0..300 x 0..80
        let strokes = vec![
            line(20.0, 10.0, 20.0, 70.0, 0.0),
            line(60.0, 70.0, 60.0, 10.0, 1.0),
            line(60.0, 10.0, 100.0, 70.0, 2.0),
            line(100.0, 70.0, 100.0, 10.0, 3.0),
        ];
        let img = render_to_image(&strokes, 64);
        img.save("/tmp/strokes_render.png").unwrap();
        eprintln!("rendu {}x{} -> /tmp/strokes_render.png", img.width(), img.height());
    }
}
