//! Inférence de reconnaissance d'écriture, 100 % locale et 100 % Rust :
//! le CRNN est chargé depuis un checkpoint PyTorch .pth (voir model.rs).

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use image::GrayImage;

use crate::model::Crnn;

/// Normalisation des pixels en entrée du modèle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Norm {
    /// 0..1, fond blanc = 1.0.
    ZeroOne,
    /// -1..1, encre noire = -1.0, fond blanc = 1.0.
    NegOneOne,
}

impl Norm {
    fn apply(self, v: f32) -> f32 {
        match self {
            Norm::ZeroOne => v,
            Norm::NegOneOne => v * 2.0 - 1.0,
        }
    }
}

/// `Send` : l'UI charge un autre modèle dans un thread d'arrière-plan et
/// renvoie le reconnaisseur via un channel.
pub trait Recognizer: Send {
    fn recognize(&mut self, img: &GrayImage) -> Result<String>;

    /// Dernière image effectivement envoyée au modèle (après redimensionnement
    /// à `input_h`), pour l'affichage debug. `None` pour le mock.
    fn last_input_image(&self) -> Option<&GrayImage> {
        None
    }

    /// Reconnaissance de trajectoires (online). Par défaut : rendu en image
    /// puis `recognize` (baseline CRNN). Un modèle de séquences (LRU/transformer
    /// sur features [1, T, F]) surchargera cette méthode.
    fn recognize_strokes(&mut self, strokes: &[crate::ink::Stroke]) -> Result<String> {
        let img = crate::ink::render_to_image(strokes, 64);
        self.recognize(&img)
    }
}

/// Décodage glouton CTC : argmax par pas de temps, dédoublonnage,
/// suppression du blank (`blank`, index 0 pour le CRNN IAM, dernier pour doctR).
pub fn ctc_greedy_decode(
    logits: &[f32],
    width: usize,
    height: usize,
    charset: &[char],
    blank: usize,
) -> String {
    let mut out = String::new();
    let mut prev = blank;
    for t in 0..width {
        let mut best = blank;
        let mut best_v = f32::MIN;
        for c in 0..height {
            let v = logits[t * height + c];
            if v > best_v {
                best_v = v;
                best = c;
            }
        }
        if best != blank && best != prev {
            if let Some(&ch) = charset.get(best) {
                out.push(ch);
            }
        }
        prev = best;
    }
    out
}

/// Décodage CTC par beam search (prefix beam) sur des log-probabilités
/// [T, C] (sortie log_softmax du modèle). Retourne les `nbest` meilleures
/// hypothèses (texte, log-probabilité CTC totale) triées par score décroissant.
/// Le beam interne garde au moins `max(beam_size, nbest)` préfixes.
pub fn ctc_beam_decode_nbest(
    logits: &[f32],
    width: usize,
    height: usize,
    charset: &[char],
    beam_size: usize,
    nbest: usize,
    blank: usize,
) -> Vec<(String, f64)> {
    use std::collections::HashMap;
    // beam: préfixe (classes) -> (proba se terminant par blank, proba non-blank)
    let beam_w = beam_size.max(nbest).max(1);
    let mut beam: HashMap<Vec<usize>, (f64, f64)> = HashMap::new();
    beam.insert(Vec::new(), (1.0, 0.0));
    for t in 0..width {
        let mut next: HashMap<Vec<usize>, (f64, f64)> = HashMap::new();
        for (prefix, &(pb, pnb)) in &beam {
            for c in 0..height {
                let lp = logits[t * height + c] as f64;
                if lp < -25.0 {
                    continue; // classe quasi impossible à ce pas de temps
                }
                let p = lp.exp();
                if c == blank {
                    let e = next.entry(prefix.clone()).or_insert((0.0, 0.0));
                    e.0 += (pb + pnb) * p;
                } else if prefix.last() == Some(&c) {
                    // c identique au dernier caractère du préfixe : s'il était
                    // précédé d'un blank (pb), il forme un nouveau caractère
                    // identique (séparé par le blank) -> on étend ; sinon il
                    // est collé au précédent et absorbé -> préfixe inchangé.
                    let mut ext = prefix.clone();
                    ext.push(c);
                    let e = next.entry(ext).or_insert((0.0, 0.0));
                    e.1 += pb * p;
                    let e = next.entry(prefix.clone()).or_insert((0.0, 0.0));
                    e.1 += pnb * p;
                } else {
                    let mut ext = prefix.clone();
                    ext.push(c);
                    let e = next.entry(ext).or_insert((0.0, 0.0));
                    e.1 += (pb + pnb) * p;
                }
            }
        }
        let mut v: Vec<(Vec<usize>, (f64, f64))> = next.into_iter().collect();
        v.sort_by(|a, b| {
            (b.1 .0 + b.1 .1)
                .partial_cmp(&(a.1 .0 + a.1 .1))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        beam = v.into_iter().take(beam_w).collect();
    }
    let mut v: Vec<(Vec<usize>, (f64, f64))> = beam.into_iter().collect();
    v.sort_by(|a, b| {
        (b.1 .0 + b.1 .1)
            .partial_cmp(&(a.1 .0 + a.1 .1))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    v.truncate(nbest.max(1));
    v.into_iter()
        .map(|(p, (pb, pnb))| {
            let text = p.iter().filter_map(|&i| charset.get(i)).collect::<String>();
            (text, pb + pnb)
        })
        .collect()
}

/// Décodage CTC beam à une seule hypothèse (meilleur préfixe).
pub fn ctc_beam_decode(
    logits: &[f32],
    width: usize,
    height: usize,
    charset: &[char],
    beam_size: usize,
    blank: usize,
) -> String {
    ctc_beam_decode_nbest(logits, width, height, charset, beam_size, 1, blank)
        .into_iter()
        .next()
        .map(|(s, _)| s)
        .unwrap_or_default()
}

/// Rescoring N-best par un lexique (shallow fusion) : pour chaque hypothèse,
/// `score = logp_ctc / longueur + lm_weight * couverture_lexicale`, et on
/// renvoie la meilleure. `logp_ctc` est une somme de log-probabilités sur T
/// (toujours négative) ; diviser par la longueur de l'hypothèse compense les
/// différences de longueur entre candidats. La couverture lexicale vaut 0..1.
pub fn rescore_nbest(nbest: &[(String, f64)], lexicon: &crate::lexicon::Lexicon, lm_weight: f64) -> String {
    let mut best_text = nbest.first().map(|(s, _)| s.clone()).unwrap_or_default();
    let mut best_score = f64::MIN;
    for (text, ctc_logp) in nbest {
        let n = text.chars().count().max(1) as f64;
        let score = ctc_logp / n + lm_weight * lexicon.coverage(text);
        if score > best_score {
            best_score = score;
            best_text = text.clone();
        }
    }
    best_text
}

/// Options de décodage CTC + rescoring par lexique.
#[derive(Clone)]
pub struct DecodeOptions {
    /// Largeur du beam CTC (1 = décodage glouton).
    pub beam_size: usize,
    /// Nombre d'hypothèses N-best rescorées par le lexique.
    pub nbest: usize,
    /// Poids du lexique dans le rescoring (0.0 = lexique désactivé).
    pub lm_weight: f64,
    /// Lexique de langue ; `None` = pas de rescoring.
    pub lexicon: Option<crate::lexicon::Lexicon>,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            beam_size: 16,
            nbest: 8,
            lm_weight: 2.0,
            lexicon: Some(crate::lexicon::Lexicon::embedded_fr()),
        }
    }
}

/// Backend de modèle supporté.
pub enum Backend {
    /// CRNN "Sachin-0001" (CNN 7 convs + BiLSTM 256) : entrée [1, 1, H, W],
    /// hauteur fixe en gardant le ratio, norm 0..1 ou -1..1, blank = index 0.
    Sachin(Crnn),
    /// CRNN doctR mobilenet_v3_large : entrée [1, 3, H, W] à taille FIXE
    /// (resize sans garder le ratio), normalisation (x - mean) / std,
    /// blank = dernier index.
    Doctr {
        model: crate::doctr::DoctrCrnn,
        mean: [f32; 3],
        std: [f32; 3],
        input_w: u32,
    },
    /// PyLaIA LaiaCRNN (ex. Teklia/pylaia-rimes) : entrée [1, 1, 128, W],
    /// hauteur fixe 128 en gardant le ratio, image INVERSÉE puis /255
    /// (fond 0, encre 1), blank = index 0 (`<ctc>`).
    Pylaia(crate::pylaia::PylaiaCrnn),
}

/// Reconnaissance via un CRNN chargé depuis un checkpoint torch.
/// Entrée : image grayscale prétraitée selon le backend. Sortie : logits [1, T, C].
pub struct CrnnRecognizer {
    backend: Backend,
    /// Index de la classe blank CTC (0 pour Sachin, num_classes-1 pour doctR).
    blank_id: usize,
    charset: Vec<char>,
    device: Device,
    input_h: u32,
    norm: Norm,
    opts: DecodeOptions,
    /// Image envoyée au modèle lors du dernier appel (après resize), pour le debug.
    last_input: Option<GrayImage>,
}

impl CrnnRecognizer {
    /// Modèle "Sachin-0001" (IAM) depuis un .pth + vocab.json, options par défaut.
    pub fn new(
        pth_path: &Path,
        vocab_path: &Path,
        num_classes: usize,
        hidden_size: usize,
        input_h: u32,
        norm: Norm,
        device: &Device,
    ) -> Result<Self> {
        Self::with_options(
            pth_path,
            vocab_path,
            num_classes,
            hidden_size,
            input_h,
            norm,
            DecodeOptions::default(),
            device,
        )
    }

    /// Modèle "Sachin-0001" avec options de décodage explicites.
    pub fn with_options(
        pth_path: &Path,
        vocab_path: &Path,
        num_classes: usize,
        hidden_size: usize,
        input_h: u32,
        norm: Norm,
        opts: DecodeOptions,
        device: &Device,
    ) -> Result<Self> {
        let model = Crnn::load(pth_path, num_classes, hidden_size, device)
            .with_context(|| format!("chargement du modèle {}", pth_path.display()))?;
        let charset = crate::model::load_charset(vocab_path)?;
        if charset.len() != num_classes {
            anyhow::bail!(
                "vocab ({} classes) != num_classes du modèle ({})",
                charset.len(),
                num_classes
            );
        }
        Ok(Self {
            backend: Backend::Sachin(model),
            blank_id: 0,
            charset,
            device: device.clone(),
            input_h,
            norm,
            opts,
            last_input: None,
        })
    }

    /// Modèle doctR (ex. crnn_mobilenet_v3_large français) depuis un
    /// pytorch_model.bin + config.json doctR.
    pub fn with_doctr(
        pth_path: &Path,
        cfg: &crate::doctr::DoctrConfig,
        hidden_size: usize,
        input_h: u32,
        input_w: u32,
        opts: DecodeOptions,
        device: &Device,
    ) -> Result<Self> {
        let num_classes = cfg.vocab.chars().count() + 1; // + blank (dernier)
        let model = crate::doctr::DoctrCrnn::load(pth_path, num_classes, hidden_size, device)
            .with_context(|| format!("chargement du modèle doctR {}", pth_path.display()))?;
        let (charset, blank_id) = crate::doctr::doctr_charset(&cfg.vocab, num_classes);
        let mean = [
            cfg.mean.first().copied().unwrap_or(0.5),
            cfg.mean.get(1).copied().unwrap_or(0.5),
            cfg.mean.get(2).copied().unwrap_or(0.5),
        ];
        let std = [
            cfg.std.first().copied().unwrap_or(1.0),
            cfg.std.get(1).copied().unwrap_or(1.0),
            cfg.std.get(2).copied().unwrap_or(1.0),
        ];
        Ok(Self {
            backend: Backend::Doctr { model, mean, std, input_w },
            blank_id,
            charset,
            device: device.clone(),
            input_h,
            norm: Norm::NegOneOne, // inutilisé pour doctR
            opts,
            last_input: None,
        })
    }

    /// Modèle PyLaIA (ex. Teklia/pylaia-rimes) depuis un weights.ckpt +
    /// syms.txt. Hauteur d'entrée fixe (128 par défaut).
    pub fn with_pylaia(
        pth_path: &Path,
        syms_path: &Path,
        input_h: u32,
        opts: DecodeOptions,
        device: &Device,
    ) -> Result<Self> {
        let (charset, blank_id) = crate::pylaia::pylaia_charset(syms_path)?;
        let num_classes = charset.len();
        let model = crate::pylaia::PylaiaCrnn::load(pth_path, num_classes, 256, 3, device)
            .with_context(|| format!("chargement du modèle PyLaIA {}", pth_path.display()))?;
        Ok(Self {
            backend: Backend::Pylaia(model),
            blank_id,
            charset,
            device: device.clone(),
            input_h,
            norm: Norm::ZeroOne, // inutilisé pour pylaia
            opts,
            last_input: None,
        })
    }

    /// Décodage d'une séquence de log-probabilités [T, C] : beam CTC N-best
    /// puis rescoring par le lexique si activé.
    fn decode(&self, flat: &[f32], t: usize, c: usize) -> String {
        if self.opts.beam_size <= 1 {
            return ctc_greedy_decode(flat, t, c, &self.charset, self.blank_id);
        }
        let nbest = ctc_beam_decode_nbest(
            flat,
            t,
            c,
            &self.charset,
            self.opts.beam_size,
            self.opts.nbest,
            self.blank_id,
        );
        match &self.opts.lexicon {
            Some(lex) if self.opts.lm_weight > 0.0 && nbest.len() > 1 => {
                rescore_nbest(&nbest, lex, self.opts.lm_weight)
            }
            _ => nbest.into_iter().next().map(|(s, _)| s).unwrap_or_default(),
        }
    }

    fn image_to_tensor(&mut self, img: &GrayImage) -> Result<Tensor> {
        match &self.backend {
            Backend::Sachin(_) => {
                // Redimensionne à hauteur fixe en gardant les proportions, comme
                // le prétraitement officiel du modèle (PIL BILINEAR + round).
                // Filtre bilinéaire (Triangle) : Lanczos3 créait des artefacts
                // de ringing sur les bords nets noir/blanc du rendu de strokes.
                let ratio = self.input_h as f32 / img.height() as f32;
                let w = ((img.width() as f32 * ratio).round() as u32).max(1);
                let resized = image::imageops::resize(
                    img,
                    w,
                    self.input_h,
                    image::imageops::FilterType::Triangle,
                );
                let mut data = vec![0f32; (w * self.input_h) as usize];
                for (i, p) in resized.pixels().enumerate() {
                    data[i] = self.norm.apply(p.0[0] as f32 / 255.0);
                }
                self.last_input = Some(resized.clone());
                Tensor::from_vec(
                    data,
                    (1, 1, self.input_h as usize, w as usize),
                    &self.device,
                )
                .context("construction du tenseur d'entrée")
            }
            Backend::Doctr { mean, std, input_w, .. } => {
                // doctR : resize à taille FIXE (H, W) sans garder le ratio
                // (c'est ainsi que le modèle a été entraîné), 3 canaux
                // (gris répliqué), normalisation (x/255 - mean) / std.
                let (w, h) = (*input_w, self.input_h);
                let resized = image::imageops::resize(
                    img,
                    w,
                    h,
                    image::imageops::FilterType::Triangle, // torchvision Resize = BILINEAR
                );
                let hw = (w * h) as usize;
                let mut data = vec![0f32; 3 * hw];
                for (i, p) in resized.pixels().enumerate() {
                    let v = p.0[0] as f32 / 255.0;
                    for c in 0..3 {
                        data[c * hw + i] = (v - mean[c]) / std[c];
                    }
                }
                self.last_input = Some(resized.clone());
                Tensor::from_vec(data, (1, 3, h as usize, w as usize), &self.device)
                    .context("construction du tenseur d'entrée")
            }
            Backend::Pylaia(_) => {
                // PyLaIA : resize hauteur fixe en gardant le ratio (LANCZOS),
                // image INVERSÉE (fond 0, encre 1) puis /255 -> [0, 1].
                let ratio = self.input_h as f32 / img.height() as f32;
                let w = ((img.width() as f32 * ratio).round() as u32).max(1);
                let resized = image::imageops::resize(
                    img,
                    w,
                    self.input_h,
                    image::imageops::FilterType::Lanczos3,
                );
                let mut data = vec![0f32; (w * self.input_h) as usize];
                for (i, p) in resized.pixels().enumerate() {
                    data[i] = (255 - p.0[0]) as f32 / 255.0;
                }
                self.last_input = Some(resized.clone());
                Tensor::from_vec(
                    data,
                    (1, 1, self.input_h as usize, w as usize),
                    &self.device,
                )
                .context("construction du tenseur d'entrée")
            }
        }
    }
}

impl Recognizer for CrnnRecognizer {
    fn recognize(&mut self, img: &GrayImage) -> Result<String> {
        let input = self.image_to_tensor(img)?;
        let logits = match &self.backend {
            Backend::Sachin(m) => m.forward(&input).context("inférence CRNN")?,
            Backend::Doctr { model, .. } => model.forward(&input).context("inférence doctR")?,
            Backend::Pylaia(m) => m.forward(&input).context("inférence PyLaIA")?,
        };
        // logits [1, T, C]
        let dims = logits.dims();
        let (t, c) = (dims[1], dims[2]);
        let flat = logits
            .flatten_all()
            .context("aplatissement des logits")?
            .to_vec1::<f32>()
            .context("conversion f32")?;
        Ok(self.decode(&flat, t, c))
    }

    fn recognize_strokes(&mut self, strokes: &[crate::ink::Stroke]) -> Result<String> {
        // Rendu à 2x la hauteur d'entrée : le resize bilinéaire final dans
        // image_to_tensor fait alors un vrai anti-aliasing (comme une image
        // scannée), au lieu d'un double rééchantillonnage dur 64 -> 32.
        let img = crate::ink::render_to_image(strokes, self.input_h * 2);
        self.recognize(&img)
    }

    fn last_input_image(&self) -> Option<&GrayImage> {
        self.last_input.as_ref()
    }
}

/// Bouchon pour faire tourner l'UI sans modèle (renvoie une chaîne factice).
pub struct MockRecognizer;

impl Recognizer for MockRecognizer {
    fn recognize(&mut self, _img: &GrayImage) -> Result<String> {
        Ok("[reconnaissance simulée]".to_string())
    }
}

/// Reconnaissance d'un chiffre isolé (0-9) par le classifieur de régression
/// logistique softmax entraîné sur MNIST (src/digits.rs, inspiré de
/// sktan888/handwriting_recognition_logistic_regression). Backend autonome,
/// sélectionnable dans le menu de l'UI : l'entrée est rendue (ou redimensionnée)
/// en 28x28.
pub struct DigitRecognizer {
    clf: crate::digits::DigitClassifier,
    /// Dernière image 28x28 envoyée au classifieur (debug UI).
    last_input: Option<GrayImage>,
}

impl DigitRecognizer {
    pub fn new(clf: crate::digits::DigitClassifier) -> Self {
        Self {
            clf,
            last_input: None,
        }
    }

    /// Classifieur embarqué (`assets/digits-lr.f32`).
    pub fn embedded() -> Self {
        Self::new(crate::digits::DigitClassifier::embedded())
    }
}

impl Recognizer for DigitRecognizer {
    fn recognize(&mut self, img: &GrayImage) -> Result<String> {
        let small = image::imageops::resize(
            img,
            crate::digits::MNIST_SIZE,
            crate::digits::MNIST_SIZE,
            image::imageops::FilterType::Triangle,
        );
        self.last_input = Some(small.clone());
        let (d, _proba, _all) = self.clf.predict(&small);
        Ok(d.to_string())
    }

    fn recognize_strokes(&mut self, strokes: &[crate::ink::Stroke]) -> Result<String> {
        let img = crate::digits::render_strokes_28(strokes);
        self.last_input = Some(img.clone());
        let (d, _proba, _all) = self.clf.predict(&img);
        Ok(d.to_string())
    }

    fn last_input_image(&self) -> Option<&GrayImage> {
        self.last_input.as_ref()
    }
}

/// Reconnaissance hybride CRNN + classifieur de chiffres isolés : quand les
/// strokes forment un seul glyphe compact (`digits::looks_like_single_glyph`),
/// le LR-MNIST prédit un chiffre ; si sa confiance softmax >= `digit_threshold`,
/// c'est le résultat, sinon on retombe sur le reconnaisseur interne (CRNN,
/// lignes de texte). Améliore la détection des chiffres isolés (dates,
/// montants, numéros) où le CRNN confond 1/l, 0/O, 5/S...
pub struct HybridRecognizer {
    inner: Box<dyn Recognizer>,
    digits: Option<crate::digits::DigitClassifier>,
    digit_threshold: f32,
}

impl HybridRecognizer {
    pub fn new(
        inner: Box<dyn Recognizer>,
        digits: Option<crate::digits::DigitClassifier>,
        digit_threshold: f32,
    ) -> Self {
        Self {
            inner,
            digits,
            digit_threshold,
        }
    }

    /// Le classifieur de chiffres est-il actif ?
    pub fn digits_active(&self) -> bool {
        self.digits.is_some()
    }

    pub fn digit_threshold(&self) -> f32 {
        self.digit_threshold
    }
}

impl Recognizer for HybridRecognizer {
    fn recognize(&mut self, img: &GrayImage) -> Result<String> {
        // Image fournie telle quelle (ligne de texte) : pas d'heuristique de
        // glyphe possible -> le reconnaisseur interne.
        self.inner.recognize(img)
    }

    fn recognize_strokes(&mut self, strokes: &[crate::ink::Stroke]) -> Result<String> {
        if let Some(clf) = &self.digits {
            if crate::digits::looks_like_single_glyph(strokes) {
                let img = crate::digits::render_strokes_28(strokes);
                let (digit, proba, _all) = clf.predict(&img);
                if proba >= self.digit_threshold {
                    return Ok(digit.to_string());
                }
            }
        }
        self.inner.recognize_strokes(strokes)
    }

    fn last_input_image(&self) -> Option<&GrayImage> {
        self.inner.last_input_image()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construit des logits [T, C] où chaque pas de temps favorise un classe.
    fn logits_from_path(path: &[usize], n_classes: usize) -> Vec<f32> {
        let mut v = vec![-10.0f32; path.len() * n_classes];
        for (t, &c) in path.iter().enumerate() {
            v[t * n_classes + c] = 10.0;
        }
        v
    }

    #[test]
    fn ctc_decode_regles_blank_et_repetition() {
        // charset: 0=blank, 1='a', 2='b'
        let charset: Vec<char> = vec!['\u{0}', 'a', 'b'];
        // [a, a] -> "a" (répétition sans blank entre = 1 seul caractère en CTC)
        let l = logits_from_path(&[1, 1, 1], 3);
        assert_eq!(ctc_greedy_decode(&l, 3, 3, &charset, 0), "a");
        // [a, blank, a] -> "aa" (le blank sépare les deux 'a')
        let l = logits_from_path(&[1, 0, 1], 3);
        assert_eq!(ctc_greedy_decode(&l, 3, 3, &charset, 0), "aa");
        // [a, b, a] -> "aba"
        let l = logits_from_path(&[1, 2, 1], 3);
        assert_eq!(ctc_greedy_decode(&l, 3, 3, &charset, 0), "aba");
        // tout blank -> ""
        let l = logits_from_path(&[0, 0, 0], 3);
        assert_eq!(ctc_greedy_decode(&l, 3, 3, &charset, 0), "");
        // [b, blank, blank, a] -> "ba"
        let l = logits_from_path(&[2, 0, 0, 1], 3);
        assert_eq!(ctc_greedy_decode(&l, 4, 3, &charset, 0), "ba");
    }

    #[test]
    fn ctc_beam_decode_meme_regles_que_greedy() {
        let charset: Vec<char> = vec!['\u{0}', 'a', 'b'];
        // [a, a] -> "a" ; [a, blank, a] -> "aa" ; [a, b, a] -> "aba"
        for (path, attendu) in [
            (vec![1usize, 1, 1], "a"),
            (vec![1, 0, 1], "aa"),
            (vec![1, 2, 1], "aba"),
            (vec![0, 0, 0], ""),
        ] {
            let l = logits_from_path(&path, 3);
            assert_eq!(ctc_beam_decode(&l, path.len(), 3, &charset, 8, 0), attendu);
        }
    }

    #[test]
    fn ctc_beam_prefere_la_sequence_la_plus_probable() {
        // Deux hypothèses proches : "ab" (fort à t0/t1, faible à t2) vs "aab".
        // Le beam doit choisir "ab".
        let charset: Vec<char> = vec!['\u{0}', 'a', 'b'];
        let t = 3;
        let c = 3;
        let mut l = vec![-10.0f32; t * c];
        // t0: a fort
        l[0 * c + 1] = 0.0;
        // t1: b fort
        l[1 * c + 2] = 0.0;
        // t2: blank
        l[2 * c + 0] = 0.0;
        let s = ctc_beam_decode(&l, t, c, &charset, 8, 0);
        assert_eq!(s, "ab");
    }

    #[test]
    fn rescore_nbest_prefere_le_mot_du_lexique() {
        let lex = crate::lexicon::Lexicon::from_text("le\nchat\nmange\n");
        // "le chot" a une meilleure log-prob CTC, mais "chat" est dans le
        // lexique : avec un poids LM suffisant, "le chat" gagne.
        let nbest = vec![
            ("le chot mange".to_string(), -4.0f64),
            ("le chat mange".to_string(), -4.5f64),
        ];
        assert_eq!(rescore_nbest(&nbest, &lex, 0.0), "le chot mange"); // LM off -> meilleur CTC
        assert_eq!(rescore_nbest(&nbest, &lex, 5.0), "le chat mange"); // LM on -> mot connu
    }

    #[test]
    fn rescore_nbest_sans_lexique_renvoie_le_top1() {
        let lex = crate::lexicon::Lexicon::from_text("");
        let nbest = vec![("abc".to_string(), -1.0), ("abd".to_string(), -2.0)];
        assert_eq!(rescore_nbest(&nbest, &lex, 10.0), "abc");
    }

    #[test]
    fn norm_neg1_bornes() {
        // blanc (255) -> +1.0, noir (0) -> -1.0, comme p/127.5 - 1 du handler officiel.
        assert_eq!(Norm::NegOneOne.apply(1.0), 1.0);
        assert_eq!(Norm::NegOneOne.apply(0.0), -1.0);
        assert_eq!(Norm::ZeroOne.apply(0.5), 0.5);
    }

    /// Stroke "1" : trait vertical isolé.
    fn stroke_vertical() -> Vec<crate::ink::Stroke> {
        use crate::ink::Point;
        let mut s = crate::ink::Stroke::new();
        for i in 0..=20 {
            let r = i as f32 / 20.0;
            s.push(Point {
                x: 50.0 + r * 4.0,
                y: 10.0 + r * 80.0,
                t: r as f64,
                pressure: 0.8,
            });
        }
        vec![s]
    }

    /// Stroke "mot" : longue ligne horizontale (plusieurs glyphes).
    fn stroke_mot() -> Vec<crate::ink::Stroke> {
        use crate::ink::Point;
        let mut s = crate::ink::Stroke::new();
        for i in 0..=40 {
            let r = i as f32 / 40.0;
            s.push(Point {
                x: 10.0 + r * 300.0,
                y: 50.0 + (r * 8.0).sin() * 5.0,
                t: r as f64,
                pressure: 0.8,
            });
        }
        vec![s]
    }

    #[test]
    fn hybride_glyphe_isole_passe_par_le_lr_mot_par_le_crnn() {
        // Seuil très bas : même avec des poids nuls (proba uniforme 0.1), le
        // LR répond un chiffre pour un glyphe isolé.
        let clf = crate::digits::DigitClassifier::embedded();
        let mut rec = HybridRecognizer::new(Box::new(MockRecognizer), Some(clf), 0.05);
        assert!(rec.digits_active());
        let out = rec.recognize_strokes(&stroke_vertical()).unwrap();
        assert_eq!(out.chars().count(), 1);
        assert!(out.chars().next().unwrap().is_ascii_digit(), "attendu un chiffre, reçu {out:?}");
        // Un "mot" (ligne longue) reste délégué au CRNN/mock.
        assert_eq!(rec.recognize_strokes(&stroke_mot()).unwrap(), "[reconnaissance simulée]");
    }

    #[test]
    fn hybride_sans_classifieur_delegue_toujours() {
        let mut rec = HybridRecognizer::new(Box::new(MockRecognizer), None, 0.85);
        assert!(!rec.digits_active());
        assert_eq!(rec.recognize_strokes(&stroke_vertical()).unwrap(), "[reconnaissance simulée]");
    }

    #[test]
    fn digit_recognizer_reconnait_echantillons_mnist() {
        let mut rec = DigitRecognizer::embedded();
        let samples = crate::digits::embedded_samples();
        assert!(samples.len() >= 50);
        let mut ok = 0;
        for (label, px) in &samples {
            let img = GrayImage::from_fn(crate::digits::MNIST_SIZE, crate::digits::MNIST_SIZE, |x, y| {
                image::Luma([px[(y * crate::digits::MNIST_SIZE + x) as usize]])
            });
            if rec.recognize(&img).unwrap() == label.to_string() {
                ok += 1;
            }
        }
        let acc = ok as f32 / samples.len() as f32;
        assert!(acc >= 0.80, "accuracy DigitRecognizer {acc} < 0.80");
        // last_input_image exposé pour le debug
        assert!(rec.last_input_image().is_some());
    }

    fn levenshtein(a: &[char], b: &[char]) -> usize {
        let mut prev: Vec<usize> = (0..=b.len()).collect();
        let mut cur = vec![0usize; b.len() + 1];
        for (i, &ca) in a.iter().enumerate() {
            cur[0] = i + 1;
            for (j, &cb) in b.iter().enumerate() {
                cur[j + 1] = (prev[j] + if ca == cb { 0 } else { 1 })
                    .min(prev[j + 1] + 1)
                    .min(cur[j] + 1);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        prev[b.len()]
    }

    /// Bench réel (à lancer explicitement) : évalue le pipeline complet
    /// (image -> resize bilinéaire h=32 -> norm [-1,1] -> CRNN -> CTC greedy)
    /// sur des échantillons IAM extraits dans /tmp/iam_samples (voir meta.json).
    /// Le modèle annonce CER ~ 0.126 sur le test set IAM-line.
    /// `cargo test -- --ignored --nocapture bench_iam`
    #[test]
    #[ignore]
    fn bench_iam_echantillons_reels() {
        let dir = std::path::Path::new("/tmp/iam_samples");
        if !dir.join("meta.json").exists() {
            eprintln!("skip: /tmp/iam_samples/meta.json absent");
            return;
        }
        let device = Device::Cpu;
        let (pth, vocab, config_json) =
            crate::model::download_model_files("Sachin-0001/crnn-iam-handwriting").unwrap();
        let mconfig: crate::model::ModelConfig =
            serde_json::from_str(&std::fs::read_to_string(config_json).unwrap()).unwrap();
        // Lexique anglais de calibration si dispo (les labels IAM sont en
        // anglais ; le lexique embarqué par défaut est français).
        let lex_en = std::path::Path::new("/tmp/lex20k.txt")
            .exists()
            .then(|| crate::lexicon::Lexicon::load(std::path::Path::new("/tmp/lex20k.txt")).unwrap());
        let mk = |opts: DecodeOptions| {
            CrnnRecognizer::with_options(
                &pth,
                &vocab,
                mconfig.num_classes.unwrap_or(80),
                mconfig.hidden_size.unwrap_or(256),
                mconfig.image_height.unwrap_or(32),
                Norm::NegOneOne,
                opts,
                &device,
            )
            .unwrap()
        };
        let mut rec_greedy = mk(DecodeOptions {
            beam_size: 1,
            nbest: 1,
            lm_weight: 0.0,
            lexicon: None,
        });
        let mut rec_beam = mk(DecodeOptions {
            beam_size: 16,
            nbest: 8,
            lm_weight: 0.0,
            lexicon: None,
        });
        // Calibration du poids LM sur plusieurs valeurs (lexique anglais, les
        // labels IAM étant en anglais).
        let weights = [0.5f64, 1.0, 2.0, 4.0, 8.0, 16.0];
        let mut recs_lm: Vec<(f64, CrnnRecognizer)> = weights
            .iter()
            .map(|&w| {
                (
                    w,
                    mk(DecodeOptions {
                        beam_size: 16,
                        nbest: 8,
                        lm_weight: w,
                        lexicon: lex_en.clone(),
                    }),
                )
            })
            .collect();
        let meta: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap())
                .unwrap();
        let mut cer_g = 0.0f64;
        let mut cer_b = 0.0f64;
        let mut cer_lm = vec![0.0f64; weights.len()];
        let mut n = 0usize;
        for m in &meta {
            let file = m["file"].as_str().unwrap();
            let label = m["label"].as_str().unwrap();
            let img = image::open(dir.join(file)).unwrap().to_luma8();
            let pred_g = rec_greedy.recognize(&img).unwrap();
            let pred_b = rec_beam.recognize(&img).unwrap();
            let lc: Vec<char> = label.chars().collect();
            let cer = |p: &str| {
                let pc: Vec<char> = p.chars().collect();
                levenshtein(&pc, &lc) as f64 / lc.len().max(1) as f64
            };
            let mut line = format!(
                "{file} greedy={:.3} beam={:.3}",
                cer(&pred_g),
                cer(&pred_b)
            );
            for (i, (_, rec)) in recs_lm.iter_mut().enumerate() {
                let p = rec.recognize(&img).unwrap();
                cer_lm[i] += cer(&p);
                line.push_str(&format!(" lm{}={:.3}", weights[i], cer(&p)));
            }
            println!("{line}\n  label: {label}\n  beam : {pred_b}");
            cer_g += cer(&pred_g);
            cer_b += cer(&pred_b);
            n += 1;
        }
        print!("CER moyen sur {n} échantillons: greedy={:.4} beam={:.4}", cer_g / n as f64, cer_b / n as f64);
        for (i, &w) in weights.iter().enumerate() {
            print!(" lm(w={w})={:.4}", cer_lm[i] / n as f64);
        }
        println!();
    }

    /// Bench FR réel (à lancer explicitement) : modèle PyLaIA français
    /// (~/modeles/pylaia-rimes/weights.ckpt, Teklia/pylaia-rimes) sur des
    /// lignes RIMES-2011 extraites dans /tmp/rimes_samples.
    /// Compare beam pur vs beam+lexique FR. Le modèle annonce CER ~4.5%.
    /// `cargo test --release -- --ignored --nocapture bench_rimes`
    #[test]
    #[ignore]
    fn bench_rimes_francais() {
        let dir = std::path::Path::new("/tmp/rimes_samples");
        let model_dir = std::path::Path::new("/home/lbonnier/modeles/pylaia-rimes");
        if !dir.join("meta.json").exists() || !model_dir.join("weights.ckpt").exists() {
            eprintln!("skip: /tmp/rimes_samples ou ~/modeles/pylaia-rimes absent");
            return;
        }
        let device = Device::Cpu;
        let mk = |lm_weight: f64| {
            CrnnRecognizer::with_pylaia(
                &model_dir.join("weights.ckpt"),
                &model_dir.join("syms.txt"),
                128,
                DecodeOptions {
                    beam_size: 16,
                    nbest: 8,
                    lm_weight,
                    lexicon: Some(crate::lexicon::Lexicon::embedded_fr()),
                },
                &device,
            )
            .unwrap()
        };
        let mut rec_beam = mk(0.0);
        let mut rec_lm = mk(2.0);
        let meta: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap())
                .unwrap();
        let mut cer_b = 0.0f64;
        let mut cer_lm = 0.0f64;
        let mut n = 0usize;
        for m in &meta {
            let file = m["file"].as_str().unwrap();
            let label = m["label"].as_str().unwrap();
            let img = image::open(dir.join(file)).unwrap().to_luma8();
            let pred_b = rec_beam.recognize(&img).unwrap();
            let pred_lm = rec_lm.recognize(&img).unwrap();
            let lc: Vec<char> = label.chars().collect();
            let cer = |p: &str| {
                let pc: Vec<char> = p.chars().collect();
                levenshtein(&pc, &lc) as f64 / lc.len().max(1) as f64
            };
            println!(
                "{file} beam={:.3} beam+lm={:.3}\n  label: {label}\n  beam : {pred_b}\n  lm   : {pred_lm}",
                cer(&pred_b),
                cer(&pred_lm),
            );
            cer_b += cer(&pred_b);
            cer_lm += cer(&pred_lm);
            n += 1;
        }
        println!(
            "CER moyen RIMES sur {n} lignes: beam={:.4} beam+lexiqueFR={:.4}",
            cer_b / n as f64,
            cer_lm / n as f64
        );
    }

    /// Demo visuelle (à lancer explicitement) : reconnaît des strokes
    /// synthétiques "IN" et sauve l'image exacte envoyée au modèle.
    /// `cargo test --release -- --ignored --nocapture dump_input_modele`
    #[test]
    #[ignore]
    fn dump_input_modele() {
        use crate::ink::{Point, Stroke};
        fn line(x0: f32, y0: f32, x1: f32, y1: f32, t0: f64) -> Stroke {
            let mut s = Stroke::new();
            for i in 0..=40 {
                let r = i as f32 / 40.0;
                s.push(Point {
                    x: x0 + r * (x1 - x0),
                    y: y0 + r * (y1 - y0),
                    t: t0 + r as f64 * 0.5,
                    pressure: 0.8,
                });
            }
            s
        }
        let strokes = vec![
            line(20.0, 10.0, 20.0, 70.0, 0.0),
            line(60.0, 70.0, 60.0, 10.0, 1.0),
            line(60.0, 10.0, 100.0, 70.0, 2.0),
            line(100.0, 70.0, 100.0, 10.0, 3.0),
        ];
        let device = Device::Cpu;
        let (pth, vocab, config_json) =
            crate::model::download_model_files("Sachin-0001/crnn-iam-handwriting").unwrap();
        let mconfig: crate::model::ModelConfig =
            serde_json::from_str(&std::fs::read_to_string(config_json).unwrap()).unwrap();
        let mut rec = CrnnRecognizer::new(
            &pth,
            &vocab,
            mconfig.num_classes.unwrap_or(80),
            mconfig.hidden_size.unwrap_or(256),
            mconfig.image_height.unwrap_or(32),
            Norm::NegOneOne,
            &device,
        )
        .unwrap();
        let pred = rec.recognize_strokes(&strokes).unwrap();
        eprintln!("prediction strokes 'IN' (bâtons): {pred:?}");
        if let Some(g) = rec.last_input_image() {
            g.save("/tmp/model_input.png").unwrap();
            eprintln!("input modèle {}x{} -> /tmp/model_input.png", g.width(), g.height());
        }
    }
}
