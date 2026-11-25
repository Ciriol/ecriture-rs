//! Convertit un checkpoint torch PyLaIA (.ckpt/.pth) en .safetensors,
//! format standard ouvert chargeable nativement par candle (et par Python,
//! llama.cpp, etc.) — pratique pour partager les poids hors Hugging Face.
//!
//! Usage :
//!   cargo run --release --example to_safetensors -- <weights.ckpt> <out.safetensors> [f16]
//!
//!   f16 (optionnel) : convertit en half-precision, divise la taille par ~2
//!   (le CRNN PyLaIA tourne très bien en f16 ; re-quantifiez en f32 si besoin
//!   de parité bit-à-bit avec PyTorch).

use std::collections::HashMap;
use std::env;

use candle_core::{Device, DType, Tensor};

fn main() -> anyhow::Result<()> {
    let src = env::args().nth(1).expect("usage: to_safetensors <in.ckpt> <out.safetensors> [f16]");
    let dst = env::args().nth(2).expect("usage: to_safetensors <in.ckpt> <out.safetensors> [f16]");
    let f16 = env::args().nth(3).as_deref() == Some("f16");
    let device = Device::Cpu;

    let mut archive = pthrs::PthArchive::open(&src)?;
    let tensors = archive.load_floating_f32()?;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    let mut total = 0usize;
    for (name, t) in tensors {
        let (shape, values) = t.into_shape_and_values()?;
        let tensor = Tensor::from_vec(values, shape, &device)?;
        let tensor = if f16 { tensor.to_dtype(DType::F16)? } else { tensor };
        total += tensor.elem_count();
        map.insert(name, tensor);
    }
    candle_core::safetensors::save(&map, &dst)?;
    let size = std::fs::metadata(&dst)?.len();
    println!(
        "converti {} tenseurs ({} paramètres) : {src} -> {dst} ({:.1} Mo, {})",
        map.len(),
        total,
        size as f64 / 1_048_576.0,
        if f16 { "f16" } else { "f32" }
    );
    Ok(())
}
