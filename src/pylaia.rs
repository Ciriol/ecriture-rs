//! Modèle PyLaIA LaiaCRNN (ex. Teklia/pylaia-rimes, français, CER ~4.5% sur RIMES),
//! chargé en pur Rust depuis un `weights.ckpt` (state_dict torch, lu via pthrs).
//!
//! Architecture (laia/models/htr/laia_crnn.py + defaults CreateCRNNArgs) :
//!   4 blocs ConvBlock : conv 3x3 pad 1 (sans biais, batchnorm=true) + BatchNorm
//!     + LeakyReLU(0.01) + MaxPool(2,2) pour les 3 premiers blocs, pas de pool au 4e
//!     -> features [1, 48, 16, W/8] pour une entrée de hauteur 128
//!   sequencer columnwise : [1, 48, 16, W'] -> [W', 1, 768]
//!   LSTM(768, 256, num_layers=3, bidirectional=True)  -> [W', 1, 512]
//!   Linear(512, num_labels=100)                         -> logits [W', 1, 100]
//!   blank CTC = index 0 (`<ctc>` dans syms.txt).
//!
//! Prétraitement (laia/data/transforms/vision/transforms.py ToImageTensor) :
//!   gris -> INVERSION (fond=0, encre=1) -> resize hauteur fixe 128 en gardant
//!   le ratio (LANCZOS) -> /255.  Entrée [1, 1, 128, W], valeurs 0..1.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{Device, DType, IndexOp, Tensor};
use candle_nn::rnn::{Direction, LSTMConfig, RNN, LSTM};
use candle_nn::{batch_norm, linear, BatchNorm, Conv2d, Conv2dConfig, Linear, Module, ModuleT, VarBuilder};

/// Charge le charset depuis un syms.txt PyLaIA ("symbole index" par ligne).
/// `<ctc>` -> '\0' (blank), `<space>` -> ' ', `<unk>` -> '\u{FFFD}'.
/// Retourne (charset, blank_id).
pub fn pylaia_charset(syms_path: &Path) -> Result<(Vec<char>, usize)> {
    let content = std::fs::read_to_string(syms_path)
        .with_context(|| format!("syms introuvable: {}", syms_path.display()))?;
    let mut items: Vec<(String, usize)> = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let sym = parts.next().unwrap_or("");
        let idx: usize = parts
            .next()
            .and_then(|v| v.parse().ok())
            .with_context(|| format!("ligne syms invalide: {line}"))?;
        items.push((sym.to_string(), idx));
    }
    let max_idx = items.iter().map(|(_, i)| *i).max().unwrap_or(0);
    let mut chars = vec!['\u{FFFD}'; max_idx + 1];
    let mut blank_id = 0usize;
    for (sym, idx) in items {
        let ch = match sym.as_str() {
            "<ctc>" => {
                blank_id = idx;
                '\u{0}'
            }
            "<space>" => ' ',
            "<unk>" => '\u{FFFD}',
            s => s.chars().next().unwrap_or('\u{FFFD}'),
        };
        chars[idx] = ch;
    }
    Ok((chars, blank_id))
}

/// CRNN PyLaIA (LaiaCRNN) : 4 blocs CNN + LSTM 3 couches bidir + linear.
pub struct PylaiaCrnn {
    blocks: Vec<(Conv2d, BatchNorm, bool)>,
    lstms_f: Vec<LSTM>,
    lstms_r: Vec<LSTM>,
    linear: Linear,
}

impl PylaiaCrnn {
    /// Charge depuis weights.ckpt. `features` = [12, 24, 48, 48], hidden = 256, layers = 3.
    pub fn load(pth_path: &Path, num_classes: usize, hidden: usize, layers: usize, device: &Device) -> Result<Self> {
        let mut archive = pthrs::PthArchive::open(pth_path)
            .with_context(|| format!("ouverture du checkpoint {}", pth_path.display()))?;
        let tensors = archive.load_floating_f32()?;
        let mut map = HashMap::new();
        for (name, t) in tensors {
            let (shape, values) = t.into_shape_and_values()?;
            // le ckpt PyLaIA préfixe tout par "model."
            let name = name.strip_prefix("model.").unwrap_or(&name).to_string();
            map.insert(name, Tensor::from_vec(values, shape, device)?);
        }
        let vb = VarBuilder::from_tensors(map, DType::F32, device);

        // 4 blocs : channels 12, 24, 48, 48 ; pool (2,2) sur les 3 premiers.
        let channels = [12usize, 24, 48, 48];
        let mut blocks = Vec::new();
        let mut in_ch = 1usize;
        for (i, &out_ch) in channels.iter().enumerate() {
            let pool = i < 3;
            let conv = Conv2d::new(
                vb.pp(format!("conv.{i}.conv")).get((out_ch, in_ch, 3, 3), "weight")?,
                None, // batchnorm=true -> pas de biais
                Conv2dConfig { padding: 1, ..Default::default() },
            );
            let bn = batch_norm(out_ch, 1e-5, vb.pp(format!("conv.{i}.batchnorm")))?;
            blocks.push((conv, bn, pool));
            in_ch = out_ch;
        }

        let mut lstms_f = Vec::new();
        let mut lstms_r = Vec::new();
        let last_ch = *channels.last().unwrap();
        for layer in 0..layers {
            let input_size = if layer == 0 { last_ch * 16 } else { 2 * hidden };
            let cfg_f = LSTMConfig { layer_idx: layer, direction: Direction::Forward, ..Default::default() };
            let cfg_r = LSTMConfig { layer_idx: layer, direction: Direction::Backward, ..Default::default() };
            lstms_f.push(LSTM::new(input_size, hidden, cfg_f, vb.pp("rnn"))?);
            lstms_r.push(LSTM::new(input_size, hidden, cfg_r, vb.pp("rnn"))?);
        }
        let linear = linear(2 * hidden, num_classes, vb.pp("linear"))?;
        Ok(Self { blocks, lstms_f, lstms_r, linear })
    }

    /// Passe les `n` premiers blocs CNN et retourne les features (debug/parité).
    pub fn features_up_to(&self, x: &Tensor, n: usize) -> Result<Tensor> {
        let mut f = x.clone();
        for (conv, bn, pool) in self.blocks.iter().take(n) {
            let y = conv.forward(&f)?;
            let y = bn.forward_t(&y, false)?;
            let y = leaky_relu(&y, 0.01)?;
            f = if *pool { y.max_pool2d(2)? } else { y };
        }
        Ok(f)
    }

    /// Entrée [1, 1, 128, W] -> logits bruts [1, W/8, num_classes].
    pub fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        let mut f = self.features_up_to(x, self.blocks.len())?;
        // sequencer : ramène la hauteur à 16 (identité si déjà 16), puis
        // [1, C, 16, W'] -> [1, W', C*16]
        let h = f.dim(2)?;
        if h != 16 {
            let k = h / 16;
            if k > 1 {
                f = f.max_pool2d_with_stride((k, 1), (k, 1))?;
            }
        }
        let (b, c, h, w) = f.dims4()?;
        let seq = f
            .permute((3, 0, 1, 2))? // [W', 1, C, H]
            .reshape((w, b, c * h))?
            .contiguous()?; // [W', 1, C*H]
        // candle RNN seq prend [batch, seq, features] : on permute en [1, W', C*H]
        let seq = seq.permute((1, 0, 2))?.contiguous()?;
        let mut s = seq;
        for (lf, lr) in self.lstms_f.iter().zip(&self.lstms_r) {
            s = bilstm_layer(&s, lf, lr)?;
        }
        let logits = self.linear.forward(&s)?; // [1, W', num_classes]
        Ok(logits)
    }

    /// Entrée [1, 1, 128, W] (valeurs 0..1, fond 0, encre 1) -> log-prob [1, W/8, num_classes].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.forward_raw(x)?;
        Ok(candle_nn::ops::log_softmax(&logits, 2)?)
    }
}

fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    // LeakyReLU: max(x, 0) + slope * min(x, 0)
    let pos = x.clamp(0.0, f32::MAX)?;
    let neg = (x.clamp(f32::MIN, 0.0)? * slope)?;
    Ok((pos + neg)?)
}

/// Couche BiLSTM, entrée [B, T, F] (candle seq = [batch, seq, features]).
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
    fn charset_syms() {
        let dir = std::env::temp_dir().join(format!("pylaia-syms-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("syms.txt");
        std::fs::write(&p, "<ctc> 0\na 1\n<space> 2\n<unk> 3\né 4\n").unwrap();
        let (chars, blank) = pylaia_charset(&p).unwrap();
        assert_eq!(blank, 0);
        assert_eq!(chars[0], '\u{0}');
        assert_eq!(chars[1], 'a');
        assert_eq!(chars[2], ' ');
        assert_eq!(chars[3], '\u{FFFD}');
        assert_eq!(chars[4], 'é');
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Parité numérique avec PyTorch/PyLaIA : logits bruts vs /tmp/pylaia_io
    /// (dumpés par /tmp/ref_pylaia.py, même entrée prétraitée).
    /// `cargo test --release -- --ignored --nocapture parite_pylaia`
    #[test]
    #[ignore]
    fn parite_pylaia_avec_pytorch() {
        let dir = std::path::Path::new("/tmp/pylaia_io");
        let model_dir = std::path::Path::new("/home/lbonnier/modeles/pylaia-rimes");
        if !dir.join("input.f32").exists() {
            eprintln!("skip: /tmp/pylaia_io absent (lancer /tmp/ref_pylaia.py)");
            return;
        }
        let device = Device::Cpu;
        let (charset, _blank) = pylaia_charset(&model_dir.join("syms.txt")).unwrap();
        let model = PylaiaCrnn::load(&model_dir.join("weights.ckpt"), charset.len(), 256, 3, &device).unwrap();

        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        let xs = meta["x"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect::<Vec<_>>();
        let raw = std::fs::read(dir.join("input.f32")).unwrap();
        let x_data: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let x = Tensor::from_vec(x_data, xs.as_slice(), &device).unwrap();
        let read_f32 = |name: &str| -> Vec<f32> {
            std::fs::read(dir.join(name))
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        for n in 1..=4 {
            let f = model.features_up_to(&x, n).unwrap();
            let ours = f.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let reff = read_f32(&format!("block{}.f32", n - 1));
            let max_diff = ours.iter().zip(&reff).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            println!("bloc CNN {n}: shape {:?} max|diff|={max_diff:.6}", f.dims());
        }
        let logits = model.forward_raw(&x).unwrap();
        let ours = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let ref_raw = std::fs::read(dir.join("logits.f32")).unwrap();
        let reff: Vec<f32> = ref_raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        assert_eq!(ours.len(), reff.len());
        let max_diff = ours.iter().zip(&reff).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("logits: {} valeurs, max|diff|={max_diff:.6}", ours.len());
        assert!(max_diff < 0.05, "écart trop grand vs PyTorch: {max_diff}");

        // decode greedy des logits Rust (log_softmax) et comparaison texte
        let lp = candle_nn::ops::log_softmax(&logits, 2).unwrap();
        let dims = lp.dims();
        let flat = lp.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let txt = crate::recognizer::ctc_greedy_decode(&flat, dims[1], dims[2], &charset, _blank);
        println!("decode greedy Rust: {txt:?}");
    }
}
