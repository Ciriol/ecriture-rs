//! Segmentation d'une ligne d'écriture manuscrite en **mots** par la méthode
//! *scale-space* de Manmatha & Srimal (1999) — mêmes idées que le dépôt
//! githubharald/WordDetector (noyau gaussien anisotrope + seuillage d'Otsu +
//! composantes connexes), porté en Rust pur : crate `image` seule, **aucun
//! poids de modèle**, 100 % local.
//!
//! Le détecteur n'est **pas scale-invariant** : on normalise la ligne à
//! [`SegParams::target_height`] (50 px par défaut, comme upstream — les mots
//! doivent faire ~25-50 px de haut). Bon quand les espaces inter-mots sont
//! larges (prise de notes) ; sous-segmente si les mots sont collés (cursif).

use anyhow::Result;
use image::{GrayImage, Luma, imageops};

use crate::ink::Stroke;
use crate::recognizer::Recognizer;

/// Paramètres du détecteur (défaut = paramètres upstream de WordDetector).
#[derive(Clone, Debug)]
pub struct SegParams {
    /// Taille du noyau de filtrage (impair).
    pub kernel_size: usize,
    /// Écart-type de la gaussienne selon y.
    pub sigma: f64,
    /// Ratio largeur/hauteur approximatif d'un mot (anisotropie selon x).
    pub theta: f64,
    /// Aire minimale d'un candidat mot (pixels de la composante connexe).
    pub min_area: u64,
    /// Hauteur de normalisation de la ligne avant détection.
    pub target_height: u32,
}

impl Default for SegParams {
    fn default() -> Self {
        Self {
            kernel_size: 25,
            sigma: 11.0,
            theta: 7.0,
            min_area: 100,
            target_height: 50,
        }
    }
}

/// Bounding box d'un mot détecté (repère image, origine en haut à gauche).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Redimensionne l'image à `height` en conservant le ratio (bilinéaire),
/// comme `prepare_img` upstream (cv2.resize par défaut = INTER_LINEAR).
pub fn prepare_img(img: &GrayImage, height: u32) -> GrayImage {
    let factor = height as f64 / img.height() as f64;
    let w = ((img.width() as f64) * factor).round().max(1.0) as u32;
    imageops::resize(img, w, height, imageops::FilterType::Triangle)
}

/// Noyau anisotrope = somme de deux dérivées secondes de gaussiennes
/// (x_term + y_term) * exp(-x²/2σx² - y²/2σy²), normalisé (somme = 1).
/// Identique à `_compute_kernel` de WordDetector.
fn compute_kernel(kernel_size: usize, sigma: f64, theta: f64) -> Vec<f64> {
    assert!(kernel_size % 2 == 1, "kernel_size doit être impair");
    let half = (kernel_size / 2) as f64;
    let n = kernel_size;
    let sigma_y = sigma;
    let sigma_x = sigma_y * theta;
    let mut k = vec![0.0f64; n * n];
    let mut sum = 0.0;
    for j in 0..n {
        for i in 0..n {
            let x = i as f64 - half;
            let y = j as f64 - half;
            let exp_term =
                (-x * x / (2.0 * sigma_x * sigma_x) - y * y / (2.0 * sigma_y * sigma_y)).exp();
            let x_term = (x * x - sigma_x * sigma_x)
                / (2.0 * std::f64::consts::PI * sigma_x.powi(5) * sigma_y);
            let y_term = (y * y - sigma_y * sigma_y)
                / (2.0 * std::f64::consts::PI * sigma_y.powi(5) * sigma_x);
            let v = (x_term + y_term) * exp_term;
            k[j * n + i] = v;
            sum += v;
        }
    }
    for v in k.iter_mut() {
        *v /= sum;
    }
    k
}

/// Convolution 2D avec bord répliqué, sortie saturée en u8 (comme
/// `cv2.filter2D` sur une image uint8 : saturate_cast, arrondi).
fn filter2d_replicate(img: &GrayImage, kernel: &[f64], ksize: usize) -> GrayImage {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let half = (ksize / 2) as i32;
    let mut out = GrayImage::new(img.width(), img.height());
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f64;
            for ky in 0..ksize as i32 {
                let sy = (y + ky - half).clamp(0, h - 1);
                for kx in 0..ksize as i32 {
                    let sx = (x + kx - half).clamp(0, w - 1);
                    acc += kernel[(ky as usize) * ksize + kx as usize]
                        * img.get_pixel(sx as u32, sy as u32)[0] as f64;
                }
            }
            let v = acc.round().clamp(0.0, 255.0) as u8;
            out.put_pixel(x as u32, y as u32, Luma([v]));
        }
    }
    out
}

/// Seuil d'Otsu (maximise la variance inter-classe sur l'histogramme 256 bins).
fn otsu_threshold(img: &GrayImage) -> u8 {
    let mut hist = [0u64; 256];
    for p in img.pixels() {
        hist[p[0] as usize] += 1;
    }
    let total: u64 = hist.iter().sum();
    let sum_total: u64 = (0..256).map(|i| i as u64 * hist[i]).sum();
    let mut sum_b = 0u64;
    let mut w_b = 0u64;
    let mut best = -1.0f64;
    let mut best_t = 0u8;
    for t in 0..256usize {
        w_b += hist[t];
        sum_b += t as u64 * hist[t];
        let w_f = total - w_b;
        if w_b == 0 || w_f == 0 {
            continue;
        }
        let m_b = sum_b as f64 / w_b as f64;
        let m_f = (sum_total - sum_b) as f64 / w_f as f64;
        let var = w_b as f64 * w_f as f64 * (m_b - m_f).powi(2);
        if var > best {
            best = var;
            best_t = t as u8;
        }
    }
    best_t
}

/// Détecte les mots d'une image de ligne grayscale (encre sombre sur fond
/// clair) et renvoie les boîtes **triées dans l'ordre de lecture**
/// (x + w/2 croissant), comme `detect` + `sort_line` upstream.
pub fn detect(img: &GrayImage, p: &SegParams) -> Vec<BBox> {
    let kernel = compute_kernel(p.kernel_size, p.sigma, p.theta);
    let filtered = filter2d_replicate(img, &kernel, p.kernel_size);
    let t = otsu_threshold(&filtered);
    // img_thres = 255 - (filtered > t ? 255 : 0)  =>  encre = filtered <= t
    let (w, h) = (img.width(), img.height());
    let fpx: Vec<u8> = filtered.pixels().map(|px| px[0]).collect();
    let bin: Vec<u8> = fpx.iter().map(|&v| if v <= t { 255 } else { 0 }).collect();

    // Composantes connexes 8-connexité (BFS), aire = nombre de pixels.
    let mut visited = vec![false; (w * h) as usize];
    let mut dets = Vec::new();
    for start in 0..(w * h) as usize {
        if visited[start] || bin[start] == 0 {
            continue;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (u32::MAX, u32::MAX, 0u32, 0u32);
        let mut area = 0u64;
        while let Some(idx) = stack.pop() {
            let (x, y) = ((idx as u32) % w, (idx as u32) / w);
            area += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let nidx = (ny as u32 * w + nx as u32) as usize;
                    if !visited[nidx] && bin[nidx] == 255 {
                        visited[nidx] = true;
                        stack.push(nidx);
                    }
                }
            }
        }
        if area >= p.min_area {
            dets.push(BBox {
                x: min_x,
                y: min_y,
                w: max_x - min_x + 1,
                h: max_y - min_y + 1,
            });
        }
    }
    dets.sort_by_key(|d| d.x + d.w / 2);
    dets
}

/// Détecte les mots d'une image de ligne déjà rendue à une certaine hauteur
/// (`img`, ex. `render_to_image(strokes, 64)`) : normalise à
/// `target_height`, détecte, puis **rapporte les boîtes dans le repère de
/// `img`** (la normalisation n'étant qu'un rescale uniforme).
pub fn detect_in_image(img: &GrayImage, p: &SegParams) -> Vec<BBox> {
    let seg = prepare_img(img, p.target_height);
    let scale = img.height() as f64 / seg.height() as f64;
    detect(&seg, p)
        .into_iter()
        .map(|b| BBox {
            x: (b.x as f64 * scale) as u32,
            y: (b.y as f64 * scale) as u32,
            w: (b.w as f64 * scale).ceil().max(1.0) as u32,
            h: (b.h as f64 * scale).ceil().max(1.0) as u32,
        })
        .collect()
}

/// Recadre `b` dans `img` avec une marge de `pad` px de chaque côté (bornes
/// serrées sur l'image) : le CRNN-IAM attend des marges autour du texte
/// (comme `render_to_image` qui met ~75 % de la hauteur + padding horizontal).
fn crop_padded(img: &GrayImage, b: BBox, pad: u32) -> GrayImage {
    let x = b.x.saturating_sub(pad);
    let y = b.y.saturating_sub(pad);
    let x2 = (b.x + b.w + pad).min(img.width());
    let y2 = (b.y + b.h + pad).min(img.height());
    let (w, h) = (x2.saturating_sub(x).max(1), y2.saturating_sub(y).max(1));
    imageops::crop_imm(img, x, y, w, h).to_image()
}

/// Reconnaisseur qui **segmente la ligne en mots** (détecteur scale-space)
/// puis reconnaît chaque mot séparément avec le reconnaisseur interne,
/// en joignant les résultats par des espaces explicites.
///
/// Si un seul mot est détecté (ou aucun), repli sur le pipeline complet de
/// l interne (reconnaissance de la ligne entière, ex. heuristique chiffres
/// isolés du `HybridRecognizer`) : même comportement qu'avant.
pub struct WordSegRecognizer {
    inner: Box<dyn Recognizer>,
    params: SegParams,
    /// Hauteur du rendu des strokes (2x `input_h` du CRNN, comme
    /// `CrnnRecognizer::recognize_strokes`).
    render_h: u32,
}

impl WordSegRecognizer {
    pub fn new(inner: Box<dyn Recognizer>, render_h: u32) -> Self {
        Self {
            inner,
            params: SegParams::default(),
            render_h,
        }
    }

    pub fn with_params(inner: Box<dyn Recognizer>, render_h: u32, params: SegParams) -> Self {
        Self {
            inner,
            params,
            render_h,
        }
    }

    /// Reconnaissance par mot à partir d'une image de ligne rendue.
    fn recognize_words(&mut self, img: &GrayImage) -> Result<String> {
        let boxes = detect_in_image(img, &self.params);
        if boxes.len() <= 1 {
            return self.inner.recognize(img);
        }
        let mut words = Vec::new();
        let pad = img.height() / 8;
        for b in boxes {
            let c = crop_padded(img, b, pad);
            let s = self.inner.recognize(&c)?;
            let s = s.trim();
            if !s.is_empty() {
                words.push(s.to_string());
            }
        }
        Ok(words.join(" "))
    }
}

impl Recognizer for WordSegRecognizer {
    fn recognize(&mut self, img: &GrayImage) -> Result<String> {
        self.recognize_words(img)
    }

    fn recognize_strokes(&mut self, strokes: &[Stroke]) -> Result<String> {
        let img = crate::ink::render_to_image(strokes, self.render_h);
        let boxes = detect_in_image(&img, &self.params);
        if boxes.len() <= 1 {
            // Un seul « mot » (ou aucun) : repli sur le pipeline complet.
            return self.inner.recognize_strokes(strokes);
        }
        let mut words = Vec::new();
        let pad = img.height() / 8;
        for b in boxes {
            let c = crop_padded(&img, b, pad);
            let s = self.inner.recognize(&c)?;
            let s = s.trim();
            if !s.is_empty() {
                words.push(s.to_string());
            }
        }
        Ok(words.join(" "))
    }

    fn last_input_image(&self) -> Option<&GrayImage> {
        self.inner.last_input_image()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ink::{Point, Stroke};

    fn draw_rect(img: &mut GrayImage, x: u32, y: u32, w: u32, h: u32) {
        for yy in y..y + h {
            for xx in x..x + w {
                img.put_pixel(xx, yy, Luma([0]));
            }
        }
    }

    /// Un « mot » synthétique : zigzag horizontal épais sur `x0..x0+width`.
    fn word_stroke(x0: f32, width: f32, y_base: f32, t0: f64) -> Stroke {
        let mut s = Stroke::new();
        for i in 0..=40 {
            let x = x0 + width * i as f32 / 40.0;
            let y = y_base + (i as f32 / 3.0).sin() * 18.0;
            s.push(Point {
                x,
                y,
                t: t0 + i as f64 * 0.02,
                pressure: 0.9,
            });
        }
        s
    }

    #[test]
    fn kernel_normalise() {
        let k = compute_kernel(25, 11.0, 7.0);
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "somme du noyau = {sum}");
    }

    #[test]
    #[should_panic(expected = "impair")]
    fn kernel_pair_refuse() {
        compute_kernel(24, 11.0, 7.0);
    }

    #[test]
    fn otsu_bimodal() {
        // 40 % des pixels à 10, le reste à 200 : le seuil sépare les deux modes.
        let mut img = GrayImage::from_pixel(100, 100, Luma([200]));
        draw_rect(&mut img, 0, 0, 40, 100);
        for yy in 0..100 {
            for xx in 0..40 {
                img.put_pixel(xx, yy, Luma([10]));
            }
        }
        let t = otsu_threshold(&img);
        assert!((10..200).contains(&t), "seuil Otsu entre les deux modes: {t}");
    }

    /// Test synthétique identique à tests/test_synthetic.py de WordDetector.
    #[test]
    fn synthetique_trois_mots() {
        let mut img = GrayImage::from_pixel(512, 512, Luma([255]));
        let gts = [(100u32, 100u32, 100u32, 25u32), (300, 110, 50, 15), (100, 300, 50, 20)];
        for &(x, y, w, h) in &gts {
            draw_rect(&mut img, x, y, w, h);
        }
        let p = SegParams {
            kernel_size: 25,
            sigma: 25.0,
            theta: 5.0,
            min_area: 100,
            target_height: 512,
        };
        let dets = detect(&img, &p);
        assert_eq!(dets.len(), 3, "detections: {dets:?}");
        // chaque GT doit correspondre à une détection (tolérance 12 px : la bbox
        // par composantes connexes est un peu plus large que le contour cv2)
        for &(gx, gy, gw, gh) in &gts {
            let ok = dets.iter().any(|d| {
                (d.x as i32 - gx as i32).abs() <= 12
                    && (d.y as i32 - gy as i32).abs() <= 12
                    && (d.w as i32 - gw as i32).abs() <= 12
                    && (d.h as i32 - gh as i32).abs() <= 12
            });
            assert!(ok, "pas de détection proche de GT ({gx},{gy},{gw},{gh}) dans {dets:?}");
        }
    }

    /// 4 mots sur un canvas 1000x50, paramètres par défaut -> 4 boîtes ordonnées.
    #[test]
    fn quatre_mots_canvas() {
        let mut img = GrayImage::from_pixel(1000, 50, Luma([255]));
        for (x, w) in [(20u32, 120u32), (220, 80), (380, 200), (660, 150)] {
            draw_rect(&mut img, x, 10, w, 30);
        }
        let dets = detect(&img, &SegParams::default());
        assert_eq!(dets.len(), 4, "detections: {dets:?}");
        let xs: Vec<u32> = dets.iter().map(|d| d.x).collect();
        assert!(xs.windows(2).all(|w| w[0] < w[1]), "tri lecture: {xs:?}");
    }

    /// Pipeline complet : strokes -> render_to_image(64) -> prepare_img(50) ->
    /// detect -> boîtes dans le repère du rendu 64. Deux mots attendus.
    #[test]
    fn pipeline_strokes_deux_mots() {
        let strokes = vec![
            word_stroke(0.0, 80.0, 40.0, 0.0),
            word_stroke(200.0, 80.0, 40.0, 1.0),
        ];
        let img = crate::ink::render_to_image(&strokes, 64);
        let boxes = detect_in_image(&img, &SegParams::default());
        assert_eq!(boxes.len(), 2, "boxes: {boxes:?} img {}x{}", img.width(), img.height());
        assert!(boxes[0].x + boxes[0].w <= boxes[1].x + 2, "mots séparés et ordonnés: {boxes:?}");
    }

    /// Mock : reconnaît chaque image par ses dimensions ; recognize_strokes = repli.
    struct MockRec;
    impl Recognizer for MockRec {
        fn recognize(&mut self, img: &GrayImage) -> Result<String> {
            Ok(format!("w{}x{}", img.width(), img.height()))
        }
        fn recognize_strokes(&mut self, _strokes: &[Stroke]) -> Result<String> {
            Ok("FALLBACK".to_string())
        }
    }

    #[test]
    fn wordseg_reco_par_mot() {
        let strokes = vec![
            word_stroke(0.0, 80.0, 40.0, 0.0),
            word_stroke(200.0, 80.0, 40.0, 1.0),
        ];
        let mut rec = WordSegRecognizer::new(Box::new(MockRec), 64);
        let out = rec.recognize_strokes(&strokes).unwrap();
        let mots: Vec<&str> = out.split(' ').collect();
        assert_eq!(mots.len(), 2, "attendu 2 mots joints par un espace: {out:?}");
        assert!(mots.iter().all(|m| m.starts_with("w")), "{out:?}");
    }

    #[test]
    fn wordseg_repli_ligne_entiere_si_un_mot() {
        let strokes = vec![word_stroke(0.0, 80.0, 40.0, 0.0)];
        let mut rec = WordSegRecognizer::new(Box::new(MockRec), 64);
        let out = rec.recognize_strokes(&strokes).unwrap();
        assert_eq!(out, "FALLBACK");
    }

    /// Test réel (#[ignore]) : modèle Sachin en cache HF. Deux mots synthétiques
    /// (scribbles) : la segmentation doit détecter 2 mots, et la reconnaissance
    /// par mot renvoie « mot1 mot2 » (un seul espace), à comparer avec la ligne
    /// entière reconnue d'un bloc.
    #[test]
    #[ignore]
    fn wordseg_reel_deux_mots() {
        use crate::recognizer::{CrnnRecognizer, Norm};

        let strokes = vec![
            word_stroke(0.0, 150.0, 40.0, 0.0),
            word_stroke(300.0, 150.0, 40.0, 5.0),
        ];
        // la segmentation voit bien 2 boîtes dans le rendu 64 px
        let img = crate::ink::render_to_image(&strokes, 64);
        let boxes = detect_in_image(&img, &SegParams::default());
        eprintln!("boîtes détectées: {boxes:?} (img {}x{})", img.width(), img.height());
        assert_eq!(boxes.len(), 2, "attendu 2 mots: {boxes:?}");

        let device = candle_core::Device::Cpu;
        let build = || {
            let (pth, vocab, config_json) =
                crate::model::download_model_files("Sachin-0001/crnn-iam-handwriting").unwrap();
            let mconfig: crate::model::ModelConfig =
                serde_json::from_str(&std::fs::read_to_string(config_json).unwrap()).unwrap();
            CrnnRecognizer::new(
                &pth,
                &vocab,
                mconfig.num_classes.unwrap_or(80),
                mconfig.hidden_size.unwrap_or(256),
                mconfig.image_height.unwrap_or(32),
                Norm::NegOneOne,
                &device,
            )
            .unwrap()
        };
        let mut plain = build();
        let ligne = plain.recognize_strokes(&strokes).unwrap();
        let mut seg = WordSegRecognizer::new(Box::new(build()), 64);
        let mots = seg.recognize_strokes(&strokes).unwrap();
        eprintln!("ligne entière: {ligne:?} | par mots: {mots:?}");
        assert_eq!(
            mots.matches(' ').count(),
            1,
            "attendu « mot1 mot2 » (un seul espace), obtenu {mots:?}"
        );
    }
}
