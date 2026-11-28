//! bench-iam : évalue le modèle CRNN sur un jeu de lignes (fichiers InkML IAM,
//! ou images + vérité terrain) en comparant la reconnaissance **ligne entière**
//! vs **segmentation en mots** (`WordSegRecognizer`, mode `--word-seg` de l'app).
//! Harnais d'évaluation inspiré de hwrt (MartinThoma/hwrt, MIT) : métriques
//! CER/WER dans `src/eval.rs`, lecteur InkML dans `src/inkml.rs`.
//!
//! Usage :
//!   cargo run --release --bin bench-iam -- <DIR> [options]
//!
//!   <DIR>              dossier de *.inkml (annotation type="truth"), ou
//!   --images <DIR>     dossier d'images (png/jpg) + GT
//!   --gt FILE.json     mode images : {"fichier.png": "vérité terrain", ...}
//!                      (sinon sidecar <image>.txt à côté de chaque image)
//!   --repo owner/name  modèle HF (défaut Sachin-0001/crnn-iam-handwriting)
//!   --beam N           largeur du beam CTC (défaut 16)
//!   --nbest N          hypothèses rescorées (défaut 8)
//!   --lm-weight F      poids du lexique (défaut 0.0 : comparaison CTC pur)
//!   --char-lm-weight F poids du LM de caractères (défaut 0.0)
//!   --lexicon PATH     lexique de rescoring (sinon aucun)
//!   --limit N          ne benchmarker que les N premiers échantillons

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::Device;
use ecriture_rs::eval::{run_bench, summarize, BenchSample};
use ecriture_rs::model::{download_model_files, ModelConfig};
use ecriture_rs::recognizer::{CrnnRecognizer, DecodeOptions, Norm};
use ecriture_rs::wordseg::WordSegRecognizer;

fn usage() -> ! {
    eprintln!(
        "usage: bench-iam <DIR> [--images] [--gt FILE.json] [--repo owner/name]\n\
         \x20         [--beam N] [--nbest N] [--lm-weight F] [--char-lm-weight F]\n\
         \x20         [--lexicon PATH] [--limit N]"
    );
    std::process::exit(2)
}

fn main() -> Result<()> {
    let mut dir: Option<PathBuf> = None;
    let mut images = false;
    let mut gt: Option<PathBuf> = None;
    let mut repo = "Sachin-0001/crnn-iam-handwriting".to_string();
    let mut beam = 16usize;
    let mut nbest = 8usize;
    let mut lm_weight = 0.0f64;
    let mut char_lm_weight = 0.0f64;
    let mut lexicon_path: Option<PathBuf> = None;
    let mut limit = usize::MAX;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--images" => images = true,
            "--gt" => gt = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--repo" => repo = args.next().unwrap_or_else(|| usage()),
            "--beam" => beam = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--nbest" => nbest = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--lm-weight" => {
                lm_weight = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage())
            }
            "--char-lm-weight" => {
                char_lm_weight = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage())
            }
            "--lexicon" => lexicon_path = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--limit" => limit = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "-h" | "--help" => usage(),
            other if dir.is_none() && !other.starts_with('-') => dir = Some(PathBuf::from(other)),
            other => {
                eprintln!("argument inconnu: {other}");
                usage();
            }
        }
    }
    let dir = dir.context("dossier manquant (voir --help)")?;

    // 1) échantillons
    let mut samples: Vec<BenchSample> = Vec::new();
    if images {
        let gt_map: std::collections::HashMap<String, String> = match &gt {
            Some(p) => serde_json::from_str(
                &std::fs::read_to_string(p).with_context(|| format!("lecture de {}", p.display()))?,
            )
            .with_context(|| format!("parse JSON de {}", p.display()))?,
            None => Default::default(),
        };
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .with_context(|| format!("lecture de {}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("png" | "jpg" | "jpeg" | "bmp")
                )
            })
            .collect();
        files.sort();
        for f in files {
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            let g = gt_map.get(&name).cloned().or_else(|| {
                std::fs::read_to_string(f.with_extension("txt"))
                    .ok()
                    .map(|s| s.trim().to_string())
            });
            let Some(g) = g else {
                eprintln!("skip {name}: pas de GT (ni --gt ni sidecar .txt)");
                continue;
            };
            let img = image::open(&f)
                .with_context(|| format!("ouverture de {}", f.display()))?
                .to_luma8();
            samples.push(BenchSample::from_image(name, img, g));
        }
    } else {
        for (name, s) in ecriture_rs::inkml::read_inkml_dir(&dir)? {
            match BenchSample::from_inkml(name.clone(), &s) {
                Some(s) => samples.push(s),
                None => eprintln!("skip {name}: pas d'annotation type=\"truth\""),
            }
        }
    }
    samples.truncate(limit);
    if samples.is_empty() {
        anyhow::bail!("aucun échantillon avec GT dans {}", dir.display());
    }
    eprintln!("{} échantillons depuis {}", samples.len(), dir.display());

    // 2) modèle : deux instances identiques (ligne entière / enveloppé word-seg)
    let device = Device::Cpu;
    let (pth, vocab, config_json) = download_model_files(&repo)?;
    let mconfig: ModelConfig = serde_json::from_str(&std::fs::read_to_string(config_json)?)?;
    let input_h = mconfig.image_height.unwrap_or(32);
    let lex = match &lexicon_path {
        Some(p) => Some(ecriture_rs::lexicon::Lexicon::load(p)?),
        None => None,
    };
    let mk = || {
        CrnnRecognizer::with_options(
            &pth,
            &vocab,
            mconfig.num_classes.unwrap_or(80),
            mconfig.hidden_size.unwrap_or(256),
            input_h,
            Norm::NegOneOne,
            DecodeOptions {
                beam_size: beam,
                nbest,
                lm_weight,
                lexicon: lex.clone(),
                char_lm_weight,
                char_lm: None,
            },
            &device,
        )
        .unwrap()
    };
    let mut line = mk();
    let mut seg = WordSegRecognizer::new(Box::new(mk()), input_h * 2);

    // 3) benchmark
    let t0 = std::time::Instant::now();
    let rows = run_bench(&samples, &mut line, &mut seg)?;
    let dt = t0.elapsed();

    for r in &rows {
        println!("{} | GT: {:?}", r.name, r.gt);
        println!(
            "  ligne   : {:?}  CER={:.3} WER={:.3}",
            r.hyp_line, r.cer_line, r.wer_line
        );
        println!(
            "  word-seg: {:?}  CER={:.3} WER={:.3}  mots détectés: {:?}/{:?}",
            r.hyp_seg, r.cer_seg, r.wer_seg, r.words_det, r.words_gt
        );
    }
    let s = summarize(&rows);
    println!("\n=== résumé ({} échantillons, {:?}) ===", s.n, dt);
    println!("CER moyen   ligne={:.4}  word-seg={:.4}", s.mean_cer_line, s.mean_cer_seg);
    println!("WER moyen   ligne={:.4}  word-seg={:.4}", s.mean_wer_line, s.mean_wer_seg);
    println!(
        "exact (CER=0)  ligne={}/{}  word-seg={}/{}",
        s.exact_line, s.n, s.exact_seg, s.n
    );
    if let (Some(d), Some(g)) = (s.mean_words_det, s.mean_words_gt) {
        println!("mots détectés en moyenne: {d:.1} (GT: {g:.1})");
    }
    Ok(())
}
