//! Classifieur de chiffres manuscrits isolés (0-9) par régression logistique
//! multinomiale (softmax) sur des images 28x28, entraîné sur MNIST.
//!
//! S'inspire de <https://github.com/sktan888/handwriting_recognition_logistic_regression>
//! (même formulation : entrée 784 pixels normalisés 0..1 fond noir, 10 classes,
//! poids `W[10][784] + b[10]`, sigmoïde/softmax) mais en pur Rust embarqué : les
//! poids (~31 Ko) sont compilés dans le binaire (`assets/digits-lr.f32`, générés
//! par `src/bin/train-digits-lr.rs`) et l'inférence n'est qu'une multiplication
//! matrice-vecteur + softmax, sans aucune dépendance ML.
//!
//! Rôle dans le pipeline : le CRNN-CTC reconnaît des lignes entières et peut
//! confondre certains chiffres (1/l, 0/O, 5/S...). Quand la fenêtre glissante
//! ne contient qu'un seul glyphe compact (`looks_like_single_glyph`), ce
//! classifieur — fiable sur les chiffres isolés (~92 % de LR pure sur MNIST,
//! > 98 % sur les chiffres bien tracés) — décide avec un seuil de confiance ;
//! sinon on retombe sur le CRNN (`HybridRecognizer` dans recognizer.rs). Il
//! existe aussi comme backend autonome `DigitRecognizer` (menu de l'UI).

use std::path::Path;

use anyhow::{bail, Context, Result};
use image::{GrayImage, Luma};

use crate::ink::{bounding_box, Stroke};

/// Taille des images MNIST (28x28 = 784 pixels).
pub const MNIST_SIZE: u32 = 28;
/// Nombre de pixels d'une image MNIST.
pub const MNIST_PX: usize = (MNIST_SIZE * MNIST_SIZE) as usize; // 784
/// Nombre de classes (chiffres 0-9).
pub const N_CLASSES: usize = 10;

/// Poids embarqués, entraînés hors-ligne sur MNIST (60k images) par
/// `cargo run --release --bin train-digits-lr -- <dossier MNIST>` :
/// `N_CLASSES * MNIST_PX` f32 little-endian (W, une ligne de 784 par classe)
/// puis `N_CLASSES` f32 (b) -> 31 400 octets.
const EMBEDDED_WEIGHTS: &[u8] = include_bytes!("../assets/digits-lr.f32");

/// Échantillons de test MNIST embarqués (tests unitaires) : u32 LE (nombre)
/// puis, par échantillon, u8 (label 0-9) + 784 u8 (pixels). Généré par
/// `train-digits-lr --dump-samples`.
const EMBEDDED_SAMPLES: &[u8] = include_bytes!("../assets/mnist-samples.bin");

/// Régression logistique multinomiale (softmax) 784 -> 10, à plat :
/// `w[k * MNIST_PX + i]` = poids du pixel `i` pour la classe `k`.
pub struct DigitClassifier {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl DigitClassifier {
    /// Classifieur embarqué dans le binaire (`assets/digits-lr.f32`).
    pub fn embedded() -> Self {
        Self::from_f32_bytes(EMBEDDED_WEIGHTS).expect("poids LR embarqués invalides")
    }

    /// Charge des poids depuis un fichier : binaire f32 LE (31 400 octets,
    /// même layout que l'embarqué) ou JSON `{"w": [[...784...] x10], "b": [...10...]}`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("lecture des poids chiffres {}", path.display()))?;
        if bytes.len() == (N_CLASSES * MNIST_PX + N_CLASSES) * 4 {
            Self::from_f32_bytes(&bytes)
        } else {
            #[derive(serde::Deserialize)]
            struct Json {
                w: Vec<Vec<f32>>,
                b: Vec<f32>,
            }
            let j: Json = serde_json::from_slice(&bytes)
                .with_context(|| format!("poids chiffres {} ni f32 brut ni JSON", path.display()))?;
            if j.w.len() != N_CLASSES || j.w.iter().any(|r| r.len() != MNIST_PX) || j.b.len() != N_CLASSES
            {
                bail!(
                    "{} : formes invalides (attendu w 10x{}, b 10)",
                    path.display(),
                    MNIST_PX
                );
            }
            Ok(Self {
                w: j.w.into_iter().flatten().collect(),
                b: j.b,
            })
        }
    }

    /// Construit depuis des f32 LE bruts : `10*784` poids (ligne par classe) + `10` biais.
    pub fn from_f32_bytes(bytes: &[u8]) -> Result<Self> {
        let n = N_CLASSES * MNIST_PX + N_CLASSES;
        if bytes.len() != n * 4 {
            bail!(
                "poids LR : {} octets, attendu {} ({} f32)",
                bytes.len(),
                n * 4,
                n
            );
        }
        let mut v = vec![0f32; n];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            v[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(Self {
            w: v[..N_CLASSES * MNIST_PX].to_vec(),
            b: v[N_CLASSES * MNIST_PX..].to_vec(),
        })
    }

    /// Sérialise en f32 LE bruts (même layout que `assets/digits-lr.f32`).
    pub fn to_f32_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity((N_CLASSES * MNIST_PX + N_CLASSES) * 4);
        for v in self.w.iter().chain(self.b.iter()) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Prédit le chiffre d'une image 28x28 (fond blanc, encre noire, comme le
    /// rendu de `render_strokes_28`). Retourne (chiffre, proba softmax,
    /// probas des 10 classes).
    pub fn predict(&self, img: &GrayImage) -> (u8, f32, [f32; N_CLASSES]) {
        assert_eq!(
            (img.width(), img.height()),
            (MNIST_SIZE, MNIST_SIZE),
            "predict attend une image 28x28 (voir predict_resized)"
        );
        // MNIST : fond noir = 0.0, encre blanche = 1.0. Notre rendu est
        // fond blanc / encre noire -> on inverse.
        let mut x = [0f32; MNIST_PX];
        for (i, p) in img.pixels().enumerate() {
            x[i] = (255 - p.0[0]) as f32 / 255.0;
        }
        let mut logits = [0f32; N_CLASSES];
        for (k, lg) in logits.iter_mut().enumerate() {
            let row = &self.w[k * MNIST_PX..(k + 1) * MNIST_PX];
            let mut s = self.b[k];
            for i in 0..MNIST_PX {
                s += row[i] * x[i];
            }
            *lg = s;
        }
        let probs = softmax(&logits);
        let (best, &p) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        (best as u8, p, probs)
    }

    /// Prédit le chiffre d'une image quelconque : redimensionnée d'abord en
    /// 28x28 carré (sans garder le ratio, comme MNIST).
    pub fn predict_resized(&self, img: &GrayImage) -> (u8, f32, [f32; N_CLASSES]) {
        let small = image::imageops::resize(img, MNIST_SIZE, MNIST_SIZE, image::imageops::FilterType::Triangle);
        self.predict(&small)
    }
}

/// Softmax stable (max soustrait) d'un vecteur de logits.
pub fn softmax(logits: &[f32; N_CLASSES]) -> [f32; N_CLASSES] {
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let mut out = [0f32; N_CLASSES];
    let mut sum = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        out[i] = (l - m).exp();
        sum += out[i];
    }
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

/// Heuristique : les strokes forment-ils un seul glyphe compact (candidat
/// chiffre isolé) plutôt qu'un mot/une ligne ? Un chiffre isolé est à peu près
/// aussi haut que large (ratio largeur/hauteur 0.02..2.2, un "1" droit étant
/// très étroit) et tient en 1 à 4 strokes ("4" = 2-3 traits). Au-delà, c'est
/// un mot -> CRNN.
pub fn looks_like_single_glyph(strokes: &[Stroke]) -> bool {
    if strokes.is_empty() || strokes.len() > 4 {
        return false;
    }
    let Some((min_x, min_y, max_x, max_y)) = bounding_box(strokes) else {
        return false;
    };
    let w = max_x - min_x;
    let h = max_y - min_y;
    if h < 8.0 {
        return false; // trop petit = bruit de capteur
    }
    let ratio = w / h;
    (0.02..=2.2).contains(&ratio)
}

/// Rend les strokes en image 28x28 fond blanc / encre noire, glyphe centré
/// dans un carré de 20x20 (marges de 4 px, comme le cadrage MNIST), trait
/// d'environ 2.2 px avec anti-aliasing linéaire.
pub fn render_strokes_28(strokes: &[Stroke]) -> GrayImage {
    let mut img = GrayImage::from_pixel(MNIST_SIZE, MNIST_SIZE, Luma([255]));
    let Some((min_x, min_y, max_x, max_y)) = bounding_box(strokes) else {
        return img;
    };
    let w = (max_x - min_x).max(1e-3);
    let h = (max_y - min_y).max(1e-3);
    let scale = 20.0 / w.max(h);
    let off_x = (MNIST_SIZE as f32 - w * scale) / 2.0 - min_x * scale;
    let off_y = (MNIST_SIZE as f32 - h * scale) / 2.0 - min_y * scale;
    let radius = 1.1;
    let map = |p: &crate::ink::Point| (p.x * scale + off_x, p.y * scale + off_y);
    for s in strokes {
        if s.points.len() == 1 {
            let (x, y) = map(&s.points[0]);
            stamp_disk(&mut img, x, y, radius);
        }
        for win in s.points.windows(2) {
            let (x0, y0) = map(&win[0]);
            let (x1, y1) = map(&win[1]);
            draw_thick_segment(&mut img, x0, y0, x1, y1, radius);
        }
    }
    img
}

/// Disque d'encre anti-aliasé (falloff linéaire sur 1 px), fond blanc 255.
fn stamp_disk(img: &mut GrayImage, cx: f32, cy: f32, r: f32) {
    let x0 = ((cx - r - 1.0).floor() as i32).max(0);
    let y0 = ((cy - r - 1.0).floor() as i32).max(0);
    let x1 = ((cx + r + 1.0).ceil() as i32).min(MNIST_SIZE as i32 - 1);
    let y1 = ((cy + r + 1.0).ceil() as i32).min(MNIST_SIZE as i32 - 1);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
            let a = (r + 0.5 - d).clamp(0.0, 1.0);
            if a > 0.0 {
                let p = img.get_pixel_mut(x as u32, y as u32);
                let v = (255.0 * (1.0 - a)) as u8;
                if v < p.0[0] {
                    p.0[0] = v;
                }
            }
        }
    }
}

/// Segment épais entre deux points, échantillonné finement pour ne pas laisser
/// de trous.
fn draw_thick_segment(img: &mut GrayImage, x0: f32, y0: f32, x1: f32, y1: f32, r: f32) {
    let d = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
    let steps = ((d / 0.4).ceil() as u32).max(1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        stamp_disk(img, x0 + t * (x1 - x0), y0 + t * (y1 - y0), r);
    }
}

/// Un jeu MNIST lu depuis des fichiers IDX : `images` = n * 784 octets,
/// `labels` = n octets (0-9).
pub struct MnistSet {
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
}

/// Lit un fichier IDX d'images (magic 2051) : retourne (pixels n*rows*cols, n, rows, cols).
pub fn read_idx_images(path: &Path) -> Result<(Vec<u8>, usize, usize, usize)> {
    let b = std::fs::read(path).with_context(|| format!("lecture {}", path.display()))?;
    if b.len() < 16 || u32::from_be_bytes([b[0], b[1], b[2], b[3]]) != 2051 {
        bail!("{} : pas un fichier IDX images (magic 2051)", path.display());
    }
    let n = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
    let rows = u32::from_be_bytes([b[8], b[9], b[10], b[11]]) as usize;
    let cols = u32::from_be_bytes([b[12], b[13], b[14], b[15]]) as usize;
    let data = &b[16..];
    if data.len() != n * rows * cols {
        bail!("{} : {} octets de pixels, attendu {}", path.display(), data.len(), n * rows * cols);
    }
    Ok((data.to_vec(), n, rows, cols))
}

/// Lit un fichier IDX de labels (magic 2049).
pub fn read_idx_labels(path: &Path) -> Result<Vec<u8>> {
    let b = std::fs::read(path).with_context(|| format!("lecture {}", path.display()))?;
    if b.len() < 8 || u32::from_be_bytes([b[0], b[1], b[2], b[3]]) != 2049 {
        bail!("{} : pas un fichier IDX labels (magic 2049)", path.display());
    }
    let n = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
    if b.len() - 8 != n {
        bail!("{} : {} labels, {} octets attendus", path.display(), n, b.len() - 8);
    }
    Ok(b[8..].to_vec())
}

/// Charge les 4 fichiers IDX MNIST d'un dossier (`train-images-idx3-ubyte`,
/// `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`).
pub fn load_mnist_dir(dir: &Path) -> Result<(MnistSet, MnistSet)> {
    let read = |img: &str, lbl: &str| -> Result<MnistSet> {
        let (images, n, rows, cols) = read_idx_images(&dir.join(img))?;
        let labels = read_idx_labels(&dir.join(lbl))?;
        if rows * cols != MNIST_PX || labels.len() != n {
            bail!("{dir:?} : dimensions incohérentes ({rows}x{cols}, {n} images, {} labels)", labels.len());
        }
        Ok(MnistSet { images, labels })
    };
    let train = read("train-images-idx3-ubyte", "train-labels-idx1-ubyte")?;
    let test = read("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")?;
    Ok((train, test))
}

/// Entraîne le softmax par mini-batch SGD (descente de gradient sur
/// l'entropie croisée), poids initialisés à zéro, LR décroissant
/// `lr0 / (1 + 0.5 * époque)`, mélange déterministe (LCG) des exemples.
/// Retourne le classifieur et l'accuracy sur le jeu de test (si fourni).
pub fn train_softmax(
    train: &MnistSet,
    test: Option<&MnistSet>,
    epochs: usize,
    lr0: f32,
    batch: usize,
    seed: u64,
) -> (DigitClassifier, f32) {
    let n = train.labels.len();
    let mut w = vec![0f32; N_CLASSES * MNIST_PX];
    let mut b = vec![0f32; N_CLASSES];
    // Prénormalise les entrées 0..1 (MNIST : fond noir 0, encre blanche 255).
    let x: Vec<f32> = train.images.iter().map(|&v| v as f32 / 255.0).collect();
    // LCG déterministe pour le shuffle.
    let mut state = seed.max(1);
    let mut rand = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut order: Vec<usize> = (0..n).collect();
    for epoch in 0..epochs {
        // Fisher-Yates avec le LCG.
        for i in (1..n).rev() {
            let j = (rand() as usize) % (i + 1);
            order.swap(i, j);
        }
        let lr = lr0 / (1.0 + 0.5 * epoch as f32);
        for chunk in order.chunks(batch) {
            let mut gw = vec![0f32; N_CLASSES * MNIST_PX];
            let mut gb = [0f32; N_CLASSES];
            for &idx in chunk {
                let img = &x[idx * MNIST_PX..(idx + 1) * MNIST_PX];
                let mut logits = [0f32; N_CLASSES];
                for k in 0..N_CLASSES {
                    let row = &w[k * MNIST_PX..(k + 1) * MNIST_PX];
                    let mut s = b[k];
                    for i in 0..MNIST_PX {
                        s += row[i] * img[i];
                    }
                    logits[k] = s;
                }
                let probs = softmax(&logits);
                let y = train.labels[idx] as usize;
                for k in 0..N_CLASSES {
                    let g = probs[k] - if k == y { 1.0 } else { 0.0 };
                    gb[k] += g;
                    let row = &mut gw[k * MNIST_PX..(k + 1) * MNIST_PX];
                    for i in 0..MNIST_PX {
                        row[i] += g * img[i];
                    }
                }
            }
            let m = chunk.len() as f32;
            for k in 0..N_CLASSES {
                b[k] -= lr * gb[k] / m;
                let row = &mut w[k * MNIST_PX..(k + 1) * MNIST_PX];
                let grow = &gw[k * MNIST_PX..(k + 1) * MNIST_PX];
                for i in 0..MNIST_PX {
                    row[i] -= lr * grow[i] / m;
                }
            }
        }
        if let Some(t) = test {
            eprintln!("époque {}/{}: accuracy test = {:.4}", epoch + 1, epochs, accuracy(&w, &b, t));
        } else {
            eprintln!("époque {}/{}", epoch + 1, epochs);
        }
    }
    let acc = test.map(|t| accuracy(&w, &b, t)).unwrap_or(0.0);
    (DigitClassifier { w, b }, acc)
}

/// Accuracy du classifieur (logits W/b à plat) sur un jeu MNIST.
pub fn accuracy(w: &[f32], b: &[f32], set: &MnistSet) -> f32 {
    let mut ok = 0usize;
    for idx in 0..set.labels.len() {
        let img: Vec<f32> = set.images[idx * MNIST_PX..(idx + 1) * MNIST_PX]
            .iter()
            .map(|&v| v as f32 / 255.0)
            .collect();
        let mut best = 0usize;
        let mut best_v = f32::MIN;
        for k in 0..N_CLASSES {
            let row = &w[k * MNIST_PX..(k + 1) * MNIST_PX];
            let mut s = b[k];
            for i in 0..MNIST_PX {
                s += row[i] * img[i];
            }
            if s > best_v {
                best_v = s;
                best = k;
            }
        }
        if best == set.labels[idx] as usize {
            ok += 1;
        }
    }
    ok as f32 / set.labels.len() as f32
}

/// Extrait `per_class` échantillons par chiffre du jeu de test et les écrit en
/// binaire (u32 LE count, puis par échantillon : u8 label + 784 u8 pixels) —
/// format de `assets/mnist-samples.bin` embarqué pour les tests.
pub fn dump_samples(test: &MnistSet, per_class: usize, out: &Path) -> Result<usize> {
    let mut out_bytes = Vec::new();
    let mut samples: Vec<(u8, &[u8])> = Vec::new();
    for digit in 0..10u8 {
        let mut taken = 0;
        for idx in 0..test.labels.len() {
            if test.labels[idx] == digit && taken < per_class {
                samples.push((digit, &test.images[idx * MNIST_PX..(idx + 1) * MNIST_PX]));
                taken += 1;
            }
        }
    }
    out_bytes.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    for (label, px) in &samples {
        out_bytes.push(*label);
        out_bytes.extend_from_slice(px);
    }
    std::fs::write(out, &out_bytes).with_context(|| format!("écriture {}", out.display()))?;
    Ok(samples.len())
}

/// Charge les échantillons embarqués `assets/mnist-samples.bin` (pixels MNIST
/// bruts : fond noir, encre blanche) et les convertit au format interne du
/// projet (fond blanc 255, encre noire 0), comme `render_strokes_28`.
pub fn embedded_samples() -> Vec<(u8, [u8; MNIST_PX])> {
    let b = EMBEDDED_SAMPLES;
    let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let mut out = Vec::with_capacity(n);
    let mut off = 4;
    for _ in 0..n {
        let label = b[off];
        let mut px = [0u8; MNIST_PX];
        for i in 0..MNIST_PX {
            px[i] = 255 - b[off + 1 + i];
        }
        out.push((label, px));
        off += 1 + MNIST_PX;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ink::Point;

    fn stroke_line(x0: f32, y0: f32, x1: f32, y1: f32) -> Stroke {
        let mut s = Stroke::new();
        for i in 0..=20 {
            let r = i as f32 / 20.0;
            s.push(Point {
                x: x0 + r * (x1 - x0),
                y: y0 + r * (y1 - y0),
                t: r as f64,
                pressure: 0.8,
            });
        }
        s
    }

    #[test]
    fn softmax_somme_a_un_et_argmax() {
        let p = softmax(&[0.0, 1.0, 2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert_eq!(p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0, 2);
        // stabilité avec de grands logits
        let p = softmax(&[1000.0, 1001.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let sum: f32 = p.iter().sum();
        assert!(p[1] > 0.7 && (sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn from_f32_bytes_taille_invalide() {
        assert!(DigitClassifier::from_f32_bytes(&[0u8; 4]).is_err());
        let clf = DigitClassifier::embedded();
        let bytes = clf.to_f32_bytes();
        assert_eq!(bytes.len(), (N_CLASSES * MNIST_PX + N_CLASSES) * 4);
        let clf2 = DigitClassifier::from_f32_bytes(&bytes).unwrap();
        assert_eq!(clf2.to_f32_bytes(), bytes);
    }

    #[test]
    fn classifieur_embarque_reconnait_les_echantillons_mnist() {
        let clf = DigitClassifier::embedded();
        let samples = embedded_samples();
        assert!(samples.len() >= 50, "échantillons embarqués insuffisants");
        let mut ok = 0;
        for (label, px) in &samples {
            let img = GrayImage::from_fn(MNIST_SIZE, MNIST_SIZE, |x, y| {
                Luma([px[(y * MNIST_SIZE + x) as usize]])
            });
            let (pred, proba, _) = clf.predict(&img);
            if pred == *label {
                ok += 1;
            }
            assert!(proba > 0.0 && proba <= 1.0);
        }
        let acc = ok as f32 / samples.len() as f32;
        eprintln!("accuracy LR embarqué sur {} échantillons MNIST embarqués: {:.3}", samples.len(), acc);
        // LR pure sur MNIST ~92 % ; marge large pour un petit échantillon.
        assert!(acc >= 0.80, "accuracy embarquée {acc} < 0.80 sur {} échantillons", samples.len());
    }

    #[test]
    fn rendu_strokes_chiffre_passe_par_le_lr() {
        // "7" européen : barre du haut, diagonale, petite barre médiane.
        let strokes = vec![
            stroke_line(20.0, 15.0, 80.0, 15.0),
            stroke_line(80.0, 15.0, 40.0, 90.0),
            stroke_line(52.0, 48.0, 72.0, 48.0),
        ];
        let clf = DigitClassifier::embedded();
        let img = render_strokes_28(&strokes);
        let (d, p, _) = clf.predict(&img);
        eprintln!("strokes '7' -> LR prédit {d} (proba {p:.3})");
        // Le pipeline strokes -> 28x28 -> LR doit donner un chiffre confiant.
        assert!(p > 0.5, "confiance trop faible {p} pour un glyphe bien tracé");
        assert!(d < 10);
    }

    #[test]
    fn glyph_isole_vs_mot() {
        // "1" : trait vertical -> glyphe isolé
        assert!(looks_like_single_glyph(&[stroke_line(50.0, 10.0, 54.0, 90.0)]));
        // "4" : 3 traits dans un carré -> glyphe isolé
        assert!(looks_like_single_glyph(&[
            stroke_line(20.0, 10.0, 60.0, 10.0),
            stroke_line(40.0, 10.0, 40.0, 80.0),
            stroke_line(20.0, 50.0, 60.0, 50.0),
        ]));
        // mot : ligne horizontale longue -> ratio > 2.2 -> pas un glyphe
        assert!(!looks_like_single_glyph(&[stroke_line(10.0, 50.0, 300.0, 55.0)]));
        // trop de strokes (mot griffonné) -> pas un glyphe
        let many: Vec<Stroke> = (0..6).map(|i| stroke_line(10.0 + i as f32 * 30.0, 10.0, 20.0 + i as f32 * 30.0, 80.0)).collect();
        assert!(!looks_like_single_glyph(&many));
        // trop petit -> bruit
        assert!(!looks_like_single_glyph(&[stroke_line(10.0, 10.0, 12.0, 14.0)]));
        assert!(!looks_like_single_glyph(&[]));
    }

    #[test]
    fn render_strokes_28_taille_et_encre() {
        let img = render_strokes_28(&[stroke_line(20.0, 10.0, 24.0, 90.0)]);
        assert_eq!((img.width(), img.height()), (MNIST_SIZE, MNIST_SIZE));
        // fond blanc aux coins, encre noire au centre
        assert_eq!(img.get_pixel(0, 0).0[0], 255);
        let center = img.get_pixel(MNIST_SIZE / 2, MNIST_SIZE / 2).0[0];
        assert!(center < 128, "centre {center} devrait être encré");
    }

    #[test]
    fn lecture_idx_magic() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        let bad = dir.join("pas-un-idx.bin");
        std::fs::write(&bad, b"hello world........").unwrap();
        assert!(read_idx_images(&bad).is_err());
        assert!(read_idx_labels(&bad).is_err());
        let _ = std::fs::remove_file(&bad);
    }

    #[test]
    fn entrainement_rapide_converge_sur_jouet() {
        // Entraînement mini sur un sous-ensemble synthétique : images "0" =
        // bord noir, "1" = centre noir -> le softmax doit les séparer.
        let mut images = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..200 {
            let mut px = vec![0u8; MNIST_PX];
            for i in 0..MNIST_SIZE {
                px[(i * MNIST_SIZE) as usize] = 255; // bord haut
                px[((MNIST_SIZE - 1) * MNIST_SIZE + i) as usize] = 255; // bas
            }
            images.extend_from_slice(&px);
            labels.push(0u8);
            let mut px = vec![0u8; MNIST_PX];
            px[(14 * MNIST_SIZE + 14) as usize] = 255; // centre
            images.extend_from_slice(&px);
            labels.push(1u8);
        }
        let set = MnistSet { images, labels };
        let (clf, acc) = train_softmax(&set, Some(&set), 5, 0.5, 32, 42);
        assert!(acc >= 0.99, "accuracy {acc} sur le jouet");
        // predict_resized accepte n'importe quelle taille
        let big = image::imageops::resize(
            &render_strokes_28(&[stroke_line(20.0, 10.0, 24.0, 90.0)]),
            56,
            56,
            image::imageops::FilterType::Triangle,
        );
        let (d, p, _) = clf.predict_resized(&big);
        assert!(d < 10 && p > 0.0);
    }
}
