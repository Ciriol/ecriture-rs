//! Entraîne le classifieur de chiffres isolés (régression logistique softmax
//! MNIST, voir src/digits.rs, inspiré de sktan888/handwriting_recognition_logistic_regression)
//! et écrit les poids embarqués par l'app.
//!
//! Usage :
//!   cargo run --release --bin train-digits-lr -- <dossier MNIST> [options]
//!
//! <dossier MNIST> contient les 4 fichiers IDX (train-images-idx3-ubyte,
//! train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte),
//! ex. le dossier `data/` de
//! https://github.com/sktan888/handwriting_recognition_logistic_regression
//!
//! Options :
//!   --epochs N        époques (défaut 20)
//!   --lr F            learning rate initial (défaut 0.5, décroît en 1/(1+0.5*e))
//!   --batch N         taille des mini-batch (défaut 128)
//!   --seed N          graine du mélange (défaut 42, déterministe)
//!   --out PATH        poids f32 LE écrits (défaut assets/digits-lr.f32)
//!   --json PATH       poids JSON écrits aussi (debug, ex. --json /tmp/digits.json)
//!   --dump-samples PATH  échantillons de test embarqués pour les tests
//!                        (défaut assets/mnist-samples.bin)
//!   --n-samples N     échantillons embarqués, répartis par chiffre (défaut 100)

use std::path::PathBuf;
use std::process::exit;

use ecriture_rs::digits;

fn usage_exit() -> ! {
    eprintln!(
        "usage: train-digits-lr <dossier MNIST IDX> [--epochs N] [--lr F] [--batch N] [--seed N]\n\
         \x20                          [--out assets/digits-lr.f32] [--json assets/digits-lr.json]\n\
         \x20                          [--dump-samples assets/mnist-samples.bin] [--n-samples 100]"
    );
    exit(2)
}

fn main() {
    let mut dir: Option<PathBuf> = None;
    let mut epochs = 20usize;
    let mut lr = 0.5f32;
    let mut batch = 128usize;
    let mut seed = 42u64;
    let mut out = PathBuf::from("assets/digits-lr.f32");
    let mut json: Option<PathBuf> = None;
    let mut dump: Option<PathBuf> = Some(PathBuf::from("assets/mnist-samples.bin"));
    let mut n_samples = 100usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--epochs" => epochs = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage_exit()),
            "--lr" => lr = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage_exit()),
            "--batch" => batch = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage_exit()),
            "--seed" => seed = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage_exit()),
            "--out" => out = PathBuf::from(args.next().unwrap_or_else(|| usage_exit())),
            "--json" => json = Some(PathBuf::from(args.next().unwrap_or_else(|| usage_exit()))),
            "--no-json" => json = None,
            "--dump-samples" => dump = Some(PathBuf::from(args.next().unwrap_or_else(|| usage_exit()))),
            "--no-dump" => dump = None,
            "--n-samples" => n_samples = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage_exit()),
            "-h" | "--help" => usage_exit(),
            other if dir.is_none() && !other.starts_with("--") => dir = Some(PathBuf::from(other)),
            other => {
                eprintln!("argument inconnu: {other}");
                usage_exit();
            }
        }
    }
    let Some(dir) = dir else { usage_exit() };
    eprintln!("chargement MNIST depuis {}", dir.display());
    let (train, test) = digits::load_mnist_dir(&dir).unwrap_or_else(|e| {
        eprintln!("erreur: {e:#}");
        exit(1)
    });
    eprintln!("train: {} images, test: {} images", train.labels.len(), test.labels.len());
    let (clf, acc) = digits::train_softmax(&train, Some(&test), epochs, lr, batch, seed);
    eprintln!("accuracy finale test MNIST: {:.4}", acc);
    std::fs::write(&out, clf.to_f32_bytes()).unwrap_or_else(|e| {
        eprintln!("écriture {} : {e}", out.display());
        exit(1)
    });
    eprintln!("poids écrits: {} ({} octets)", out.display(), clf.to_f32_bytes().len());
    if let Some(j) = &json {
        let n = digits::N_CLASSES * digits::MNIST_PX;
        let w: Vec<Vec<f32>> = clf
            .to_f32_bytes()
            .chunks_exact(4)
            .take(n)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>()
            .chunks(digits::MNIST_PX)
            .map(|r| r.to_vec())
            .collect();
        // b est embarqué dans le binaire : on le relit via embedded? Non -> on
        // reconstruit depuis les octets produits par to_f32_bytes (w puis b).
        let all = clf.to_f32_bytes();
        let b: Vec<f32> = all
            .chunks_exact(4)
            .skip(n)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let txt = serde_json::to_string(&serde_json::json!({ "w": w, "b": b })).unwrap();
        std::fs::write(j, txt).unwrap();
        eprintln!("poids JSON écrits: {}", j.display());
    }
    if let Some(d) = &dump {
        let per_class = (n_samples / 10).max(1);
        let n = digits::dump_samples(&test, per_class, d).unwrap();
        eprintln!("{n} échantillons de test embarqués écrits: {}", d.display());
    }
}
