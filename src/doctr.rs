//! Modèle CRNN doctR `crnn_mobilenet_v3_large` (ex. Felix92/doctr-torch-crnn-mobilenet-v3-large-french),
//! chargé en pur Rust depuis un `pytorch_model.bin` (state_dict torch.save, lu via pthrs).
//!
//! Architecture (doctr v0.3.1, `doctr/models/recognition/crnn/pytorch.py`) :
//!   feat_extractor = torchvision mobilenet_v3_large.features  (17 blocs, sortie [B, 960, 1, W/32])
//!   seq = features.reshape(B, C*H, W).transpose(1, 2)           ([B, W, 960])
//!   decoder = nn.LSTM(960, 128, num_layers=2, bidirectional=True, batch_first=True)
//!   linear  = nn.Linear(256, len(vocab)+1)                      (blank CTC = DERNIER index)
//!
//! Prétraitement doctR : resize à (H, W) fixe du `input_shape` (sans garder le ratio),
//! 3 canaux (gris répliqué), normalisation (x/255 - mean) / std du config.json.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{Device, DType, IndexOp, Tensor};
use candle_nn::rnn::{Direction, LSTMConfig, RNN, LSTM};
use candle_nn::{batch_norm, linear, BatchNorm, Conv2d, Conv2dConfig, Linear, Module, ModuleT, VarBuilder};
use serde::Deserialize;

/// config.json d'un modèle doctR (champs utiles).
#[derive(Deserialize, Clone, Debug)]
pub struct DoctrConfig {
    #[serde(default)]
    pub arch: Option<String>,
    /// vocabulaire : un caractère par classe, dans l'ordre ; le blank CTC est
    /// le dernier index (len(vocab)).
    pub vocab: String,
    #[serde(default = "default_mean")]
    pub mean: Vec<f32>,
    #[serde(default = "default_std")]
    pub std: Vec<f32>,
    #[serde(default = "default_input_shape")]
    pub input_shape: Vec<usize>,
}

fn default_mean() -> Vec<f32> {
    vec![0.5, 0.5, 0.5]
}
fn default_std() -> Vec<f32> {
    vec![1.0, 1.0, 1.0]
}
fn default_input_shape() -> Vec<usize> {
    vec![3, 32, 128]
}

/// Charset doctR : `chars[i]` = i-ème caractère du vocab, plus une case blank
/// à la fin (dernier index = len(vocab)). Retourne (charset, blank_id).
pub fn doctr_charset(vocab: &str, num_classes: usize) -> (Vec<char>, usize) {
    let mut chars: Vec<char> = vocab.chars().collect();
    let blank_id = chars.len();
    chars.push('\u{0}');
    chars.resize(num_classes, '\u{FFFD}');
    (chars, blank_id)
}

#[derive(Clone, Copy)]
enum Act {
    Hardswish,
    Relu,
    Identity,
}

fn apply_act(x: &Tensor, act: Act) -> Result<Tensor> {
    match act {
        Act::Relu => Ok(x.relu()?),
        Act::Identity => Ok(x.clone()),
        Act::Hardswish => {
            // x * clamp(x + 3, 0, 6) / 6
            let clamped = (x + 3.0)?.clamp(0.0, 6.0)?;
            Ok(((x * clamped)? / 6.0)?)
        }
    }
}

fn hard_sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok(((x + 3.0)?.clamp(0.0, 6.0)? / 6.0)?)
}

/// Conv 2D sans biais + BatchNorm (eps 1e-5, inférence) + activation.
struct ConvBn {
    conv: Conv2d,
    bn: BatchNorm,
    act: Act,
}

impl ConvBn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?;
        let y = self.bn.forward_t(&y, false)?;
        apply_act(&y, self.act)
    }
}

/// Depthwise conv (groups = C) : poids brut [C, 1, k, k], via reshape
/// [1, C, H, W] -> [C, 1, H, W] + conv2d standard + retour, puis BN + act.
struct DwBn {
    w: Tensor,
    bn: BatchNorm,
    act: Act,
    stride: usize,
    pad: usize,
}

impl DwBn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Depthwise (groups = C) : une conv kxk par canal, puis concat.
        let (_b, c, _h, _w) = x.dims4()?;
        let cfg = Conv2dConfig { stride: self.stride, padding: self.pad, ..Default::default() };
        let mut outs = Vec::with_capacity(c);
        for i in 0..c {
            let xi = x.i((.., i..i + 1, .., ..))?.contiguous()?; // [B, 1, H, W]
            let wi = self.w.i((i..i + 1, .., .., ..))?; // [1, 1, k, k]
            let conv = Conv2d::new(wi, None, cfg);
            outs.push(conv.forward(&xi)?); // [B, 1, H', W']
        }
        let y = Tensor::cat(&outs, 1)?; // [B, C, H', W']
        let y = self.bn.forward_t(&y, false)?;
        apply_act(&y, self.act)
    }
}

/// Squeeze-Excitation (torchvision) : avgpool HxW -> fc1 1x1 + ReLU -> fc2 1x1 + HardSigmoid.
struct Se {
    fc1: Conv2d,
    fc2: Conv2d,
}

impl Se {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let scale = x.mean_keepdim(vec![2, 3])?; // [B, C, 1, 1]
        let scale = self.fc1.forward(&scale)?.relu()?;
        let scale = self.fc2.forward(&scale)?;
        let scale = hard_sigmoid(&scale)?;
        Ok(x.broadcast_mul(&scale)?)
    }
}

/// InvertedResidual (bloc bottleneck inversé de MobileNetV3).
struct IrBlock {
    expand: Option<ConvBn>,
    dw: DwBn,
    se: Option<Se>,
    project: ConvBn,
    stride: usize,
}

impl IrBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = match &self.expand {
            Some(e) => e.forward(x)?,
            None => x.clone(),
        };
        y = self.dw.forward(&y)?;
        if let Some(se) = &self.se {
            y = se.forward(&y)?;
        }
        y = self.project.forward(&y)?; // project : activation Identity
        // residual si même shape et stride 1
        if self.stride == 1 && y.shape() == x.shape() {
            Ok((x + y)?)
        } else {
            Ok(y)
        }
    }
}

enum Block {
    ConvBn(ConvBn),
    Ir(IrBlock),
}

/// CRNN doctR mobilenet_v3_large.
pub struct DoctrCrnn {
    blocks: Vec<Block>,
    l0f: LSTM,
    l0r: LSTM,
    l1f: LSTM,
    l1r: LSTM,
    linear: Linear,
}

/// Config des 16 blocs InvertedResidual de mobilenet_v3_large.features
/// (torchvision) : (in, out, kernel, stride, expand, use_se, act_relu).
const MBV3_LARGE: [(usize, usize, usize, usize, usize, bool, bool); 15] = [
    (16, 16, 3, 1, 16, false, true),
    (16, 24, 3, 2, 64, false, true),
    (24, 24, 3, 1, 72, false, true),
    (24, 40, 5, 2, 72, true, true),   // blocs 1..6 = ReLU (torchvision)
    (40, 40, 5, 1, 120, true, true),
    (40, 40, 5, 1, 120, true, true),
    (40, 80, 3, 2, 240, false, false), // blocs 7..15 = HardSwish
    (80, 80, 3, 1, 200, false, false),
    (80, 80, 3, 1, 184, false, false),
    (80, 80, 3, 1, 184, false, false),
    (80, 112, 3, 1, 480, true, false),
    (112, 112, 3, 1, 672, true, false),
    (112, 160, 5, 2, 672, true, false),
    (160, 160, 5, 1, 960, true, false),
    (160, 160, 5, 1, 960, true, false),
];

/// arrondi au multiple de 8 supérieur (torchvision _make_divisible(v, 8))
fn make_divisible(v: f32) -> usize {
    ((v / 8.0).ceil() as usize) * 8
}

impl DoctrCrnn {
    /// Charge les poids depuis un checkpoint torch (.bin/.pth), num_classes = len(vocab)+1.
    pub fn load(pth_path: &Path, num_classes: usize, hidden: usize, device: &Device) -> Result<Self> {
        let mut archive = pthrs::PthArchive::open(pth_path)
            .with_context(|| format!("ouverture du checkpoint {}", pth_path.display()))?;
        let tensors = archive.load_floating_f32()?;
        let mut map = HashMap::new();
        for (name, t) in tensors {
            let (shape, values) = t.into_shape_and_values()?;
            map.insert(name, Tensor::from_vec(values, shape, device)?);
        }
        let vb = VarBuilder::from_tensors(map, DType::F32, device);

        // Bloc 0 : ConvBNAct(3 -> 16, k3, s2, HardSwish), conv sans biais.
        let conv0 = Conv2d::new(
            vb.pp("feat_extractor.0.0").get((16, 3, 3, 3), "weight")?,
            None,
            Conv2dConfig { stride: 2, padding: 1, ..Default::default() },
        );
        let bn0 = batch_norm(16, 1e-3, vb.pp("feat_extractor.0.1"))?;
        let mut blocks = vec![Block::ConvBn(ConvBn { conv: conv0, bn: bn0, act: Act::Hardswish })];

        for (i, &(cin, cout, k, s, exp, use_se, relu)) in MBV3_LARGE.iter().enumerate() {
            let idx = i + 1;
            let vp = vb.pp(format!("feat_extractor.{idx}.block"));
            let act = if relu { Act::Relu } else { Act::Hardswish };
            let pad = k / 2;
            // expand 1x1 (présent seulement si exp != cin) : conv sans biais + BN + act
            let expand = if exp != cin {
                let w = vp.pp("0.0").get((exp, cin, 1, 1), "weight")?;
                let conv = Conv2d::new(w, None, Conv2dConfig::default());
                let bn = batch_norm(exp, 1e-3, vp.pp("0.1"))?;
                Some(ConvBn { conv, bn, act })
            } else {
                None
            };
            // Indexation torchvision de InvertedResidual.block :
            // expand (1x1) en "0" seulement si exp != cin, puis dw, puis SE, puis project.
            let has_expand = expand.is_some();
            let (dw_w_p, dw_bn_p) = if has_expand { ("1.0", "1.1") } else { ("0.0", "0.1") };
            let dw_w = vp.pp(dw_w_p).get((exp, 1, k, k), "weight")?;
            let dw_bn = batch_norm(exp, 1e-3, vp.pp(dw_bn_p))?;
            let dw = DwBn { w: dw_w, bn: dw_bn, act, stride: s, pad };
            // SE éventuel : en "2" si expand présent, "1" sinon
            let se = if use_se {
                let se_p = if has_expand { "2" } else { "1" };
                let sq = make_divisible(exp as f32 * 0.25);
                let fc1 = Conv2d::new(
                    vp.pp(format!("{se_p}.fc1")).get((sq, exp, 1, 1), "weight")?,
                    Some(vp.pp(format!("{se_p}.fc1")).get(sq, "bias")?),
                    Conv2dConfig::default(),
                );
                let fc2 = Conv2d::new(
                    vp.pp(format!("{se_p}.fc2")).get((exp, sq, 1, 1), "weight")?,
                    Some(vp.pp(format!("{se_p}.fc2")).get(exp, "bias")?),
                    Conv2dConfig::default(),
                );
                Some(Se { fc1, fc2 })
            } else {
                None
            };
            // project 1x1 : conv sans biais + BN + Identity
            let proj_idx = match (has_expand, use_se) {
                (true, true) => "3",
                (true, false) => "2",
                (false, true) => "2",
                (false, false) => "1",
            };
            let (proj_w_p, proj_bn_p) = (format!("{proj_idx}.0"), format!("{proj_idx}.1"));
            let proj_w = vp.pp(proj_w_p).get((cout, exp, 1, 1), "weight")?;
            let proj_conv = Conv2d::new(proj_w, None, Conv2dConfig::default());
            let proj_bn = batch_norm(cout, 1e-3, vp.pp(proj_bn_p))?;
            let project = ConvBn { conv: proj_conv, bn: proj_bn, act: Act::Identity };
            blocks.push(Block::Ir(IrBlock { expand, dw, se, project, stride: s }));
        }

        // Bloc 16 : ConvBNAct(160 -> 960, k1, s1, HardSwish)
        let conv16 = Conv2d::new(
            vb.pp("feat_extractor.16.0").get((960, 160, 1, 1), "weight")?,
            None,
            Conv2dConfig::default(),
        );
        let bn16 = batch_norm(960, 1e-3, vb.pp("feat_extractor.16.1"))?;
        blocks.push(Block::ConvBn(ConvBn { conv: conv16, bn: bn16, act: Act::Hardswish }));

        let lstm_cfg = |layer_idx: usize, direction: Direction| LSTMConfig {
            layer_idx,
            direction,
            ..Default::default()
        };
        let l0f = LSTM::new(960, hidden, lstm_cfg(0, Direction::Forward), vb.pp("decoder"))?;
        let l0r = LSTM::new(960, hidden, lstm_cfg(0, Direction::Backward), vb.pp("decoder"))?;
        let l1f = LSTM::new(2 * hidden, hidden, lstm_cfg(1, Direction::Forward), vb.pp("decoder"))?;
        let l1r = LSTM::new(2 * hidden, hidden, lstm_cfg(1, Direction::Backward), vb.pp("decoder"))?;
        let linear = linear(2 * hidden, num_classes, vb.pp("linear"))?;

        Ok(Self { blocks, l0f, l0r, l1f, l1r, linear })
    }

    /// Passe les `n` premiers blocs et retourne les features intermédiaires
    /// (debug / parité avec PyTorch).
    pub fn features_up_to(&self, x: &Tensor, n: usize) -> Result<Tensor> {
        let mut f = x.clone();
        for b in self.blocks.iter().take(n) {
            f = match b {
                Block::ConvBn(c) => c.forward(&f)?,
                Block::Ir(ir) => ir.forward(&f)?,
            };
        }
        Ok(f)
    }

    /// Features CNN complètes [1, 960, 1, W/32].
    pub fn features(&self, x: &Tensor) -> Result<Tensor> {
        self.features_up_to(x, self.blocks.len())
    }

    /// Entrée [1, 3, H, W] (H=32, W=128 typiquement) -> logits bruts [1, W/32, num_classes].
    pub fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        let f = self.features(x)?;
        // [1, C, H, W] -> [1, W, C*H]  (H vaut 1 en pratique : 32/32)
        let (b, c, h, w) = f.dims4()?;
        let seq = f.reshape((b, c * h, w))?.transpose(1, 2)?.contiguous()?;
        let seq = bilstm_layer(&seq, &self.l0f, &self.l0r)?;
        let seq = bilstm_layer(&seq, &self.l1f, &self.l1r)?;
        Ok(self.linear.forward(&seq)?) // [1, W', num_classes]
    }

    /// Entrée [1, 3, H, W] -> log-prob [1, W/32, num_classes].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.forward_raw(x)?;
        Ok(candle_nn::ops::log_softmax(&logits, 2)?)
    }
}

/// Couche BiLSTM (forward + backward sur séquence inversée), entrée [B, T, F].
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
    fn charset_doctr_blank_dernier() {
        let (chars, blank) = doctr_charset("abé", 4);
        assert_eq!(blank, 3);
        assert_eq!(chars, vec!['a', 'b', 'é', '\u{0}']);
    }

    #[test]
    fn make_divisible_torchvision() {
        assert_eq!(make_divisible(18.0), 24); // 72/4 -> 24 (multiple de 8)
        assert_eq!(make_divisible(120.0), 120);
        assert_eq!(make_divisible(168.0), 168);
    }

    #[test]
    fn primitives_candle() {
        let device = Device::Cpu;
        // mean_keepdim doit garder les dims : [1,2,2,2] -> [1,2,1,1]
        let x = Tensor::arange(0f32, 8f32, &device).unwrap().reshape((1, 2, 2, 2)).unwrap();
        let m = x.mean_keepdim(vec![2, 3]).unwrap();
        assert_eq!(m.dims(), &[1, 2, 1, 1]);
        let mv = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((mv[0] - 1.5).abs() < 1e-6 && (mv[1] - 5.5).abs() < 1e-6);
        // hardsigmoid: clamp(x+3,0,6)/6
        let xs = Tensor::new(&[-3f32, -1., 0., 3., 10.], &device).unwrap();
        let hs = hard_sigmoid(&xs).unwrap().to_vec1::<f32>().unwrap();
        assert!((hs[0] - 0.0).abs() < 1e-6 && (hs[1] - 2.0 / 6.0).abs() < 1e-6 && (hs[3] - 1.0).abs() < 1e-6);
        // hardswish: x*clamp(x+3,0,6)/6
        let hsw = apply_act(&xs, Act::Hardswish).unwrap().to_vec1::<f32>().unwrap();
        assert!((hsw[1] - (-1.0 * 2.0 / 6.0)).abs() < 1e-6 && (hsw[4] - 10.0).abs() < 1e-6);
    }

    /// Parité numérique avec PyTorch/doctR : compare les logits bruts de notre
    /// forward avec ceux dumpés par /tmp/dump_io.py (même entrée prétraitée).
    /// `cargo test --release -- --ignored --nocapture parite_doctr`
    #[test]
    #[ignore]
    fn parite_doctr_avec_pytorch() {
        let dir = std::path::Path::new("/tmp/doctr_io");
        let model_dir = std::path::Path::new("/home/lbonnier/modeles/crnn-fr");
        if !dir.join("input.f32").exists() {
            eprintln!("skip: /tmp/doctr_io absent (lancer /tmp/dump_io.py)");
            return;
        }
        let device = Device::Cpu;
        let cfg: DoctrConfig =
            serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap())
                .unwrap();
        let num_classes = cfg.vocab.chars().count() + 1;
        let model = DoctrCrnn::load(&model_dir.join("pytorch_model.bin"), num_classes, 128, &device)
            .unwrap();

        let raw = std::fs::read(dir.join("input.f32")).unwrap();
        let x_data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let x = Tensor::from_vec(x_data, (1, 3, 32, 128), &device).unwrap();

        // parité bloc par bloc sur les features CNN
        let read_f32 = |name: &str| -> Vec<f32> {
            std::fs::read(dir.join(name))
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        for n in 1..=model.blocks.len() {
            let f = model.features_up_to(&x, n).unwrap();
            let ours = f.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let reff = read_f32(&format!("block{:02}.f32", n - 1));
            if ours.len() != reff.len() {
                println!("bloc {n}: TAILLE DIFF ({} vs {})", ours.len(), reff.len());
                continue;
            }
            let max_diff = ours
                .iter()
                .zip(&reff)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("bloc {n}: shape {:?} max|diff|={max_diff:.6}", f.dims());
            if max_diff > 0.05 {
                println!("  >>> DIVERGENCE au bloc {n}");
                break;
            }
        }

        let logits = model.forward_raw(&x).unwrap();

        let ref_raw = std::fs::read(dir.join("logits.f32")).unwrap();
        let ref_logits: Vec<f32> = ref_raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let ours = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(ours.len(), ref_logits.len());
        let max_diff = ours
            .iter()
            .zip(&ref_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_ref = ref_logits.iter().map(|v| v.abs()).sum::<f32>() / ref_logits.len() as f32;
        println!(
            "logits: {} valeurs, max|diff|={max_diff:.6}, mean|ref|={mean_ref:.4}",
            ours.len()
        );
        assert!(max_diff < 0.05, "écart trop grand vs PyTorch: {max_diff}");
    }
}
