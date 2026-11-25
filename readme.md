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
rendu en image grayscale avec marges + lissage (src/ink.rs)
        │
        ▼
CRNN CNN-BiLSTM-CTC en candle, poids chargés depuis un .pth (src/model.rs, pthrs)
        │
        ▼
beam search CTC N-best (src/recognizer.rs) + rescoring par lexique de langue
(src/lexicon.rs, lexique français ~49k mots embarqué dans assets/) → texte affiché
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
| `--input-h N` | `image_height` du config.json (32) | hauteur de redimensionnement de l'image d'entrée. Toute valeur ≥ 32 fonctionne avec le CRNN (pool vertical adaptatif en fin de CNN) ; 32 = défaut du modèle IAM |
| `--norm 01\|neg1` | `neg1` | normalisation des pixels : `01` = 0..1 (blanc=1), `neg1` = -1..1 |
| `--tablet-device /dev/input/eventN` | auto-détection | périphérique tablette forcé (ou env `ECRITURE_TABLET_DEVICE`) |
| `--lexicon PATH\|none` | lexique français embarqué (`assets/lexique-fr.txt`) | fichier "1 mot par ligne" pour le rescoring N-best ; `none` désactive le lexique |
| `--beam N` | `16` | largeur du beam search CTC (`1` = décodage glouton) |
| `--nbest N` | `8` | nombre d'hypothèses N-best rescorées par le lexique |
| `--lm-weight F` | `2.0` | poids du lexique : `score = logp_ctc/len + F × couverture_lexicale` (`0` = désactivé) |

Dans l'UI : tracer dans la zone centrale, **Reconnaître** (tout relire), **Effacer**, case **debug** (événements d'entrée reçus).

Exemple avec un modèle local :

```bash
cargo run --release -- --pth ~/modeles/crnn/model.pth --input-h 32 --norm neg1 --tablet-device /dev/input/event9
```

## Architecture

```text
ecriture-rs/
├── Cargo.toml
├── assets/
│   └── lexique-fr.txt  lexique français embarqué (~49k mots, fréquences corpus)
├── src/
│   ├── main.rs         UI egui, capture souris/touch, debounce, options CLI
│   ├── ink.rs          Point/Stroke, resample, lissage, rendu image grayscale avec marges
│   ├── lexicon.rs      lexique de langue + couverture lexicale d'une hypothèse
│   ├── tablet.rs       lecture evdev du stylet (thread + channel + réveil egui)
│   ├── model.rs        CRNN candle-nn, chargement .pth (pthrs), téléchargement HF (hf-hub)
│   └── recognizer.rs   trait Recognizer, CrnnRecognizer, beam CTC N-best + rescoring lexique
└── LICENSE.md
```

## Tests

```bash
cargo test                          # tests unitaires (rendu, CTC greedy/beam, vocab)
cargo test -- --ignored --nocapture # test réel : télécharge le modèle HF et infère
# bench CER sur 10 vraies lignes IAM (extraites dans /tmp/iam_samples, voir meta.json) :
cargo test --release -- --ignored --nocapture bench_iam
```

Le prétraitement colle au handler officiel du modèle : rendu des strokes en image
avec marges (la bounding box occupe 75 % de la hauteur, largeur min 2×hauteur),
trait ~hauteur/10, lissage du bruit capteur, puis resize **bilinéaire** à
`--input-h` (32) et normalisation `p/127.5 - 1` (équivalent `--norm neg1`).
Décodage CTC par **beam search N-best** (`--beam`, `--nbest`) puis **rescoring par
lexique** (`--lm-weight`, lexique français embarqué par défaut). Sur 10 échantillons
IAM réels (lexique anglais de calibration) : CER glouton 0.1025 → beam+lexique
**0.0968** (le modèle annonce 0.126 sur tout son test set).

## Roadmap

- **Modèle de trajectoires (online)** : `ink::strokes_to_features` produit déjà les features `[1, T, 8]` (x, y, dx, dy, dt, pression, cos/sin) ; `Recognizer::recognize_strokes` existe (fallback image CRNN pour l'instant). Reste à brancher un LRU/petit transformer exporté en ONNX (`[1, T, F]`), inféré en pur Rust via **rten** (ou candle), avec état récurrent entre les appels. Le CRNN image reste la baseline mesurable (latence + CER).
- ~~**Fenêtre glissante** avec hypothèses révisables~~ fait : les strokes non finalisés sont re-reconnus au debounce (0,6 s) et remplacés (hypothèse en gris) ; figés après 2 s de pause ou au-delà de 12 strokes.
- **Segmentation** en mots/gestes (double-tap = espace, geste = retour à la ligne) via un petit classifieur séparé.
- Quantifier le modèle (int8) et viser < 50 ms par fenêtre ; valider le CER sur tes propres strokes.

## Licence

[GPL v2](LICENSE.md)
