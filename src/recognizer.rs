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
    model: candle_onnx::onnx::ModelProto,
    input_name: String,
    output_name: String,
    charset: Vec<char>,
    device: Device,
    input_h: u32,
}

impl CandleRecognizer {
    pub fn new(onnx_path: &str, charset_path: &str, input_h: u32) -> Result<Self> {
        let model = candle_onnx::read_file(onnx_path)
            .with_context(|| format!("modèle ONNX introuvable: {}", onnx_path))?;
        let graph = model
            .graph
            .as_ref()
            .context("le fichier ONNX ne contient pas de graphe")?;
        let input_name = graph
            .input
            .first()
            .context("le graphe ONNX n'a pas d'entrée")?
            .name
            .clone();
        let output_name = graph
            .output
            .first()
            .context("le graphe ONNX n'a pas de sortie")?
            .name
            .clone();
        let charset = load_charset(charset_path)?;
        Ok(Self {
            model,
            input_name,
            output_name,
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
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(self.input_name.clone(), input);
        let outputs = candle_onnx::simple_eval(&self.model, inputs).context("inférence ONNX")?;
        let logits = outputs
            .get(&self.output_name)
            .context("sortie du modèle introuvable")?
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
