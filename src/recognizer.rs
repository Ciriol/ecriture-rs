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

pub trait Recognizer {
    fn recognize(&mut self, img: &GrayImage) -> Result<String>;

    /// Dernière image effectivement envoyée au modèle (après redimensionnement
    /// à `input_h`), pour l'affichage debug. `None` pour le mock.
    fn last_input_image(&self) -> Option<&GrayImage> {
        None
    }
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

/// Reconnaissance via le CRNN chargé depuis un .pth (CNN-BiLSTM-CTC).
/// Entrée : image grayscale redimensionnée à hauteur `input_h`, normalisée
/// selon `norm`, tenseur [1, 1, H, W]. Sortie du modèle : logits [1, T, C].
pub struct CrnnRecognizer {
    model: Crnn,
    charset: Vec<char>,
    device: Device,
    input_h: u32,
    norm: Norm,
    /// Image envoyée au modèle lors du dernier appel (après resize), pour le debug.
    last_input: Option<GrayImage>,
}

impl CrnnRecognizer {
    pub fn new(
        pth_path: &Path,
        vocab_path: &Path,
        num_classes: usize,
        hidden_size: usize,
        input_h: u32,
        norm: Norm,
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
            model,
            charset,
            device: device.clone(),
            input_h,
            norm,
            last_input: None,
        })
    }

    fn image_to_tensor(&mut self, img: &GrayImage) -> Result<Tensor> {
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
            data[i] = self.norm.apply(p.0[0] as f32 / 255.0);
        }
        self.last_input = Some(resized);
        Tensor::from_vec(
            data,
            (1, 1, self.input_h as usize, w as usize),
            &self.device,
        )
        .context("construction du tenseur d'entrée")
    }
}

impl Recognizer for CrnnRecognizer {
    fn recognize(&mut self, img: &GrayImage) -> Result<String> {
        let input = self.image_to_tensor(img)?;
        let logits = self.model.forward(&input).context("inférence CRNN")?;
        // logits [1, T, C]
        let dims = logits.dims();
        let (t, c) = (dims[1], dims[2]);
        let flat = logits
            .flatten_all()
            .context("aplatissement des logits")?
            .to_vec1::<f32>()
            .context("conversion f32")?;
        Ok(ctc_greedy_decode(&flat, t, c, &self.charset))
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
