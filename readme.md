# ecriture-rs

[![Rust](https://img.shields.io/badge/rust-1.99+-orange?logo=rust)](https://www.rust-lang.org/)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-blue)](LICENSE.md)

Prise de notes avec **reconnaissance d'écriture manuscrite quasi temps réel, 100 % Rust et 100 % locale** — aucun Python, aucun ONNX, aucun cloud.

Écrivez à la souris, au doigt ou au **stylet** sur PC-tablette Linux ; le texte reconnu s'affiche en direct sous le canvas.

## Fonctionnement

```text
strokes (egui souris/touch  |  evdev pour le stylet /dev/input)
        │
        ▼
rendu en image grayscale (src/ink.rs)
        │
        ▼
CRNN CNN-BiLSTM-CTC en candle, poids chargés depuis un .pth (src/model.rs, pthrs)
        │
        ▼
décodage CTC glouton (src/recognizer.rs) → texte affiché
```

- Reconnaissance **incrémentale** : 0,6 s après la fin d'un tracé, seuls les nouveaux strokes sont reconnus et ajoutés au texte.
- **Stylet Linux** : lu directement via evdev (`src/tablet.rs`) quand winit ne remonte rien ; contact détecté par `BTN_TOUCH`/pression, coordonnées normalisées depuis les bornes du périphérique, pression `ABS_PRESSURE` conservée dans chaque point.
- **Modèle** : par défaut `Sachin-0001/crnn-iam-handwriting` (CRNN-CTC entraîné sur IAM), téléchargé automatiquement depuis Hugging Face (`hf-hub`) et lu en pur Rust (`pthrs`). Inférence CPU via candle.

## Prérequis

- Rust (rustup), Linux pour le support stylet.
- Pour le stylet : droits de lecture sur `/dev/input` → `sudo usermod -aG input $USER` puis **re-login**.

## Lancer

```bash
cargo run --release
```

Le modèle est téléchargé au premier lancement (cache Hugging Face). Sans modèle, l'app démarre avec un reconnaisseur simulé.

## Utilisation

| Option CLI | Défaut | Rôle |
|---|---|---|
| `--repo owner/name` | `Sachin-0001/crnn-iam-handwriting` | dépôt Hugging Face du modèle (model.pth, vocab.json, config.json) |
| `--pth PATH` | — | checkpoint `.pth` local (vocab.json et config.json attendus à côté) |
| `--input-h N` | `image_height` du config.json (32) | hauteur de l'image d'entrée du modèle |
| `--norm 01\|neg1` | `neg1` | normalisation des pixels : `01` = 0..1 (blanc=1), `neg1` = -1..1 |
| `--tablet-device /dev/input/eventN` | auto-détection | périphérique tablette forcé (ou env `ECRITURE_TABLET_DEVICE`) |

Dans l'UI : tracer dans la zone centrale, **Reconnaître** (tout relire), **Effacer**, case **debug** (événements d'entrée reçus).

Exemple avec un modèle local :

```bash
cargo run --release -- --pth ~/modeles/crnn/model.pth --input-h 32 --norm neg1 --tablet-device /dev/input/event9
```

## Architecture

```text
ecriture-rs/
├── Cargo.toml
├── src/
│   ├── main.rs         UI egui, capture souris/touch, debounce, options CLI
│   ├── ink.rs          Point/Stroke, resample, rendu image grayscale
│   ├── tablet.rs       lecture evdev du stylet (thread + channel + réveil egui)
│   ├── model.rs        CRNN candle-nn, chargement .pth (pthrs), téléchargement HF (hf-hub)
│   └── recognizer.rs   trait Recognizer, CrnnRecognizer, décodage CTC, MockRecognizer
└── LICENSE.md
```

## Tests

```bash
cargo test                          # tests unitaires
cargo test -- --ignored --nocapture # test réel : télécharge le modèle HF et infère
```

## Roadmap

- Modèle **stroke-based** (séquences x, y, t, pression) au lieu du rendu image, pour un vrai temps réel.
- **Fenêtre glissante** avec hypothèses révisables : le texte se corrige en direct.
- **Segmentation** en mots/gestes (double-tap = espace, geste = retour à la ligne).
- Valider la qualité de reconnaissance du CRNN IAM sur de l'écriture réelle.

## Licence

[GPL v2](LICENSE.md)
