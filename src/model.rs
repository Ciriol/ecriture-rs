//! Modèle CRNN (CNN + BiLSTM + classifieur linéaire + CTC) chargé directement
//! depuis un checkpoint PyTorch `.pth`, lu en pur Rust via `pthrs`.
//! Aucun Python ni ONNX requis ; les poids sont téléchargés depuis
//! Hugging Face via `hf-hub` quand aucun chemin local n'est fourni.
//!
//! Architecture : celle de `model.py` du dépôt HF (ex. Sachin-0001/crnn-iam-handwriting).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{Device, DType, Tensor};
use candle_nn::rnn::{Direction, LSTMConfig, RNN, LSTM};
use candle_nn::{
    batch_norm, conv2d, linear, BatchNorm, Conv2d, Conv2dConfig, Linear, Module, ModuleT,
    VarBuilder,
};
use serde::Deserialize;

/// config.json du modèle. Format "Sachin-0001" (IAM) : num_classes,
/// hidden_size, image_height + vocab.json à côté. Format doctR (ex. modèle
/// français Felix92/...-french) : arch + vocab (string) + mean/std/input_shape,
/// poids dans pytorch_model.bin, blank CTC = dernier index.
#[derive(Deserialize, Default, Clone, Debug)]
pub struct ModelConfig {
    #[serde(default)]
    pub num_classes: Option<usize>,
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub image_height: Option<u32>,
    #[serde(default)]
    pub arch: Option<String>,
    #[serde(default)]
    pub vocab: Option<String>,
    #[serde(default)]
    pub mean: Option<Vec<f32>>,
    #[serde(default)]
    pub std: Option<Vec<f32>>,
    #[serde(default)]
    pub input_shape: Option<Vec<usize>>,
}

impl ModelConfig {
    /// true si c'est un modèle doctR (champ `arch` ou `vocab` présent).
    pub fn is_doctr(&self) -> bool {
        self.arch.is_some() || self.vocab.is_some()
    }
}

/// vocab.json du dépôt HF.
#[derive(Deserialize)]
struct Vocab {
    char_to_idx: HashMap<String, usize>,
    blank_id: usize,
}

/// Charge le charset depuis vocab.json : `chars[i]` = caractère de la classe `i`
/// (l'index du modèle mappe directement sur le vecteur). Le blank CTC
/// (`blank_id`, 0 dans la config) occupe sa propre case. Les trous éventuels
/// entre les index du vocab sont remplis par un caractère de remplacement pour
/// ne jamais décaler le mapping classe -> caractère.
pub fn load_charset(vocab_path: &Path) -> Result<Vec<char>> {
    let content = std::fs::read_to_string(vocab_path)
        .with_context(|| format!("vocab introuvable: {}", vocab_path.display()))?;
    let vocab: Vocab = serde_json::from_str(&content).context("vocab.json invalide")?;
    let blank_id = vocab.blank_id;
    let max_idx = vocab.char_to_idx.values().copied().max().unwrap_or(0).max(blank_id);
    let mut chars = vec!['\u{FFFD}'; max_idx + 1];
    chars[blank_id] = '\u{0}'; // case du blank CTC
    for (c, &i) in &vocab.char_to_idx {
        if let Some(ch) = c.chars().next() {
            chars[i] = ch;
        }
    }
    Ok(chars)
}

/// Télécharge les fichiers d'un dépôt HF modèle. Deux formats :
/// - "Sachin-0001" (IAM) : model.pth + vocab.json + config.json
/// - doctR (ex. modèle français) : pytorch_model.bin + config.json (le vocab
///   est dans le config.json ; on renvoie config.json comme "vocab").
pub fn download_model_files(repo: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let client = hf_hub::HFClientSync::new().context("client Hugging Face")?;
    let (owner, name) = hf_hub::split_id(repo);
    let repo_hf = client.model(owner, name);
    let config = repo_hf
        .download_file()
        .filename("config.json")
        .send()
        .with_context(|| format!("téléchargement config.json depuis {repo}"))?;
    let cfg_text = std::fs::read_to_string(&config).unwrap_or_default();
    let cfg: ModelConfig = serde_json::from_str(&cfg_text).unwrap_or_default();
    if cfg.is_doctr() {
        let pth = repo_hf
            .download_file()
            .filename("pytorch_model.bin")
            .send()
            .with_context(|| format!("téléchargement pytorch_model.bin depuis {repo}"))?;
        Ok((pth, config.clone(), config))
    } else {
        let pth = repo_hf
            .download_file()
            .filename("model.pth")
            .send()
            .with_context(|| format!("téléchargement model.pth depuis {repo}"))?;
        let vocab = repo_hf
            .download_file()
            .filename("vocab.json")
            .send()
            .with_context(|| format!("téléchargement vocab.json depuis {repo}"))?;
        Ok((pth, vocab, config))
    }
}

/// CRNN : CNN (7 couches) -> BiLSTM 2 couches -> Linear -> log_softmax.
pub struct Crnn {
    conv0: Conv2d,
    conv3: Conv2d,
    conv6: Conv2d,
    bn6: BatchNorm,
    conv10: Conv2d,
    bn10: BatchNorm,
    conv14: Conv2d,
    bn14: BatchNorm,
    l0f: LSTM,
    l0r: LSTM,
    l1f: LSTM,
    l1r: LSTM,
    classifier: Linear,
}

fn pad1() -> Conv2dConfig {
    Conv2dConfig {
        padding: 1,
        ..Default::default()
    }
}

impl Crnn {
    /// Charge les poids depuis un checkpoint .pth (state_dict).
    pub fn load(pth_path: &Path, num_classes: usize, hidden: usize, device: &Device) -> Result<Self> {
        let mut archive = pthrs::PthArchive::open(pth_path)
            .with_context(|| format!("ouverture du checkpoint {}", pth_path.display()))?;
        let tensors = archive
            .load_floating_f32()
            .context("lecture des tenseurs flottants du .pth")?;
        let mut map = HashMap::new();
        for (name, t) in tensors {
            let (shape, values) = t.into_shape_and_values()?;
            map.insert(name, Tensor::from_vec(values, shape, device)?);
        }
        let vb = VarBuilder::from_tensors(map, DType::F32, device);

        let conv0 = conv2d(1, 64, 3, pad1(), vb.pp("cnn.0"))?;
        let conv3 = conv2d(64, 128, 3, pad1(), vb.pp("cnn.3"))?;
        let conv6 = conv2d(128, 256, 3, pad1(), vb.pp("cnn.6"))?;
        let bn6 = batch_norm(256, 1e-5, vb.pp("cnn.7"))?;
        let conv10 = conv2d(256, 256, 3, pad1(), vb.pp("cnn.10"))?;
        let bn10 = batch_norm(256, 1e-5, vb.pp("cnn.11"))?;
        // kernel (2, 1) non carré : conv2d() ne fait que des carrés, on construit à la main.
        let w14 = vb.pp("cnn.14").get((512, 256, 2, 1), "weight")?;
        let b14 = vb.pp("cnn.14").get(512, "bias")?;
        let conv14 = Conv2d::new(w14, Some(b14), Conv2dConfig::default());
        let bn14 = batch_norm(512, 1e-5, vb.pp("cnn.15"))?;

        let lstm_cfg = |layer_idx: usize, direction: Direction| LSTMConfig {
            layer_idx,
            direction,
            ..Default::default()
        };
        let l0f = LSTM::new(512, hidden, lstm_cfg(0, Direction::Forward), vb.pp("rnn"))?;
        let l0r = LSTM::new(512, hidden, lstm_cfg(0, Direction::Backward), vb.pp("rnn"))?;
        let l1f = LSTM::new(2 * hidden, hidden, lstm_cfg(1, Direction::Forward), vb.pp("rnn"))?;
        let l1r = LSTM::new(2 * hidden, hidden, lstm_cfg(1, Direction::Backward), vb.pp("rnn"))?;
        let classifier = linear(2 * hidden, num_classes, vb.pp("classifier"))?;

        Ok(Self {
            conv0,
            conv3,
            conv6,
            bn6,
            conv10,
            bn10,
            conv14,
            bn14,
            l0f,
            l0r,
            l1f,
            l1r,
            classifier,
        })
    }

    /// Entrée [N, 1, H, W] -> logits log_softmax [N, T, num_classes]
    /// (identique au forward PyTorch, que notre décode CTC attend en [1, T, C]).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv0.forward(x)?.relu()?;
        x = x.max_pool2d(2)?;
        x = self.conv3.forward(&x)?.relu()?;
        x = x.max_pool2d(2)?;
        x = self.conv6.forward(&x)?;
        x = self.bn6.forward_t(&x, false)?.relu()?;
        x = x.max_pool2d_with_stride((2, 1), (2, 1))?;
        x = self.conv10.forward(&x)?;
        x = self.bn10.forward_t(&x, false)?.relu()?;
        x = x.max_pool2d_with_stride((2, 1), (2, 1))?;
        x = self.conv14.forward(&x)?;
        x = self.bn14.forward_t(&x, false)?.relu()?;
        // Ramène la hauteur à 1 (pool adaptatif : toute hauteur d'entrée >= 32
        // fonctionne, pas seulement 32 — utile avec --input-h).
        let h = x.dim(2)?;
        if h > 1 {
            x = x.max_pool2d_with_stride((h, 1), (h, 1))?;
        }
        // [N, 512, 1, W] -> [N, W, 512]
        let x = x.squeeze(2)?.permute((0, 2, 1))?.contiguous()?;
        let x = bilstm_layer(&x, &self.l0f, &self.l0r)?;
        let x = bilstm_layer(&x, &self.l1f, &self.l1r)?;
        let x = self.classifier.forward(&x)?; // [N, W, C]
        Ok(candle_nn::ops::log_softmax(&x, 2)?)
    }
}

/// Une couche de BiLSTM : forward + backward (séquence inversée puis remise à l'endroit).
fn bilstm_layer(x: &Tensor, fwd: &LSTM, bwd: &LSTM) -> Result<Tensor> {
    let out_f = fwd.states_to_tensor(&fwd.seq(x)?)?;
    let x_rev = x.flip(&[1])?.contiguous()?;
    let out_b = bwd.states_to_tensor(&bwd.seq(&x_rev)?)?.flip(&[1])?;
    Ok(Tensor::cat(&[&out_f, &out_b], 2)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_charset_index_direct_et_trous() {
        // vocab avec un trou (index 3 absent) : chars[3] doit rester un
        // placeholder, pas recevoir le caractère d'index 4.
        let dir = std::env::temp_dir().join(format!("ecriture-test-vocab-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("vocab.json");
        std::fs::write(
            &p,
            r#"{"char_to_idx": {"a": 1, "b": 2, "z": 4}, "blank_id": 0}"#,
        )
        .unwrap();
        let chars = load_charset(&p).unwrap();
        assert_eq!(chars.len(), 5);
        assert_eq!(chars[0], '\u{0}');
        assert_eq!(chars[1], 'a');
        assert_eq!(chars[2], 'b');
        assert_eq!(chars[3], '\u{FFFD}');
        assert_eq!(chars[4], 'z');
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Test réel (à lancer explicitement) : télécharge le modèle HF et fait
    /// une inférence sur une image synthétique.
    /// `cargo test -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn crnn_inference_sur_image_synthetique() {
        let device = Device::Cpu;
        let (pth, vocab, config_json) =
            download_model_files("Sachin-0001/crnn-iam-handwriting").unwrap();
        let mconfig: ModelConfig =
            serde_json::from_str(&std::fs::read_to_string(config_json).unwrap()).unwrap();
        let charset = load_charset(&vocab).unwrap();
        assert_eq!(charset.len(), mconfig.num_classes.unwrap_or(80));
        let model = Crnn::load(&pth, mconfig.num_classes.unwrap_or(80), mconfig.hidden_size.unwrap_or(256), &device).unwrap();
        // image blanche 32x128 normalisée [-1, 1] -> 1.0
        let img = Tensor::ones((1, 1, mconfig.image_height.unwrap_or(32) as usize, 128), DType::F32, &device)
            .unwrap();
        let logits = model.forward(&img).unwrap();
        let dims = logits.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[2], mconfig.num_classes.unwrap_or(80));
        println!("logits shape: {:?}", dims);

        // --input-h 48 : le pool vertical adaptatif doit accepter H=48
        let img48 = Tensor::ones((1, 1, 48, 128), DType::F32, &device).unwrap();
        let logits48 = model.forward(&img48).unwrap();
        assert_eq!(logits48.dims()[2], mconfig.num_classes.unwrap_or(80));
        println!("logits shape H=48: {:?}", logits48.dims());
    }
}
