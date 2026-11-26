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
- **Modèles** : 3 backends CRNN-CTC, tous chargés en pur Rust depuis un checkpoint torch (`pthrs`), inférence CPU via candle :
  - `Sachin-0001/crnn-iam-handwriting` (défaut, anglais, IAM) : `model.pth` + `vocab.json` + `config.json`
  - **PyLaIA `Teklia/pylaia-rimes` (français, CER ~4.5% sur RIMES)** : `weights.ckpt` + `syms.txt` — lancer avec `--repo Teklia/pylaia-rimes` (téléchargement auto HF) ou `--pth ~/modeles/pylaia-rimes/weights.ckpt` (détection auto via `syms.txt`)
  - doctR `crnn_mobilenet_v3_large` : `pytorch_model.bin` + `config.json` (détection auto via `arch`/`vocab` dans le config)
  Les 3 backends sont validés bit-à-bit contre PyTorch (tests `parite_*` : logits max|diff| < 6e-5).

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
| `--repo owner/name` | `Sachin-0001/crnn-iam-handwriting` | dépôt Hugging Face du modèle, téléchargé automatiquement (cache HF). Format détecté : `syms.txt` dans le dépôt → PyLaIA (`--repo Teklia/pylaia-rimes`, FR), `arch`/`vocab` dans config.json → doctR, sinon Sachin (`model.pth` + `vocab.json` + `config.json`) |
| `--pth PATH` | — | checkpoint local ; le format est détecté depuis les fichiers à côté : `syms.txt` → PyLaIA (ex. `--pth ~/modeles/pylaia-rimes/weights.ckpt`), `arch`/`vocab` dans config.json → doctR, sinon Sachin (`model.pth` + `vocab.json` + `config.json`) |
| `--weights-url BASE` | — | partage des poids PyLaIA hors HF : télécharge `BASE/weights.safetensors` (puis `weights.ckpt` en secours) + `BASE/syms.txt` dans `~/.cache/ecriture-rs/pylaia-rimes/` au 1er lancement (GitHub Release, S3, serveur perso...) |
| `--weights-dir DIR` | — | dossier local contenant `weights.safetensors`/`weights.ckpt` + `syms.txt` |
| *(auto)* | — | sans option de modèle, cherche `weights.*` + `syms.txt` à côté du binaire, dans `./` et dans `~/.cache/ecriture-rs/pylaia-rimes/` → permet de distribuer un zip `ecriture-rs` + poids |

Priorité de résolution : `--pth` > `--weights-url` > `--weights-dir` > auto-local > `--repo` (HF).
| `--input-h N` | `image_height` du config.json (32) | hauteur de redimensionnement de l'image d'entrée. Toute valeur ≥ 32 fonctionne avec le CRNN (pool vertical adaptatif en fin de CNN) ; 32 = défaut du modèle IAM |
| `--norm 01\|neg1` | `neg1` | normalisation des pixels : `01` = 0..1 (blanc=1), `neg1` = -1..1 |
| `--tablet-device /dev/input/eventN` | auto-détection | périphérique tablette forcé (ou env `ECRITURE_TABLET_DEVICE`) |
| `--lexicon PATH\|none` | lexique français embarqué (`assets/lexique-fr.txt`) | fichier "1 mot par ligne" pour le rescoring N-best ; `none` désactive le lexique |
| `--beam N` | `16` | largeur du beam search CTC (`1` = décodage glouton) |
| `--nbest N` | `8` | nombre d'hypothèses N-best rescorées par le lexique |
| `--lm-weight F` | `2.0` | poids du lexique : `score = logp_ctc/len + F × couverture_lexicale` (`0` = désactivé) |

Dans l'UI : tracer dans la zone centrale, **Reconnaître** (tout relire), **Effacer**, **Exporter InkML** (sauve les tracés dans `exports/ink-<date>.inkml`, format W3C InkML : un `<trace>` par stroke, channels X/Y/T/P + `<annotation type="truth">` avec le label), case **debug** (événements d'entrée reçus). Le champ **vérité terrain** (au-dessus du texte reconnu) est pré-rempli avec le texte reconnu et reste éditable : corrigez-le avant d'exporter pour constituer une base de tracés online (entraînement d'un modèle de trajectoires). En haut à droite, le **menu déroulant indique le modèle actif** (source, backend, taille d'entrée) et permet d'en **changer à chaud** — le chargement (téléchargement HF + parsing du checkpoint) se fait en arrière-plan, l'UI affiche « chargement… » puis re-reconnaît le texte à l'écran avec le nouveau modèle. Choix proposés : `Sachin-0001/crnn-iam-handwriting` (IAM anglais), `Teklia/pylaia-rimes` (PyLaIA français), `Felix92/doctr-torch-crnn-mobilenet-v3-large-french` (doctR français), plus la source passée en CLI si c'est un modèle local. Le panneau du **texte reconnu en bas est redimensionnable** (tirer sa bordure haute, 120–500 px, 200 px par défaut) : texte finalisé en 28 pt, hypothèse de la fenêtre glissante en gris en dessous.

Exemples avec un modèle local :

```bash
# modèle français PyLaIA (RIMES) : téléchargé automatiquement depuis
# https://huggingface.co/Teklia/pylaia-rimes (weights.ckpt + syms.txt, cache HF)
cargo run --release -- --repo Teklia/pylaia-rimes

# idem, avec les fichiers locaux déjà téléchargés dans ~/modeles/pylaia-rimes/ :
cargo run --release -- --pth ~/modeles/pylaia-rimes/weights.ckpt

# partage des poids PyLaIA hors HF (modèle non distribué sur HF en Rust) :
# 1) convertir en safetensors (42 Mo -> 20 Mo f32, 10 Mo f16) :
cargo run --release --example to_safetensors -- ~/modeles/pylaia-rimes/weights.ckpt weights.safetensors f16
# 2) distribuer un zip avec le binaire + weights.safetensors + syms.txt
#    (détectés automatiquement à côté de l'exécutable), ou les héberger :
cargo run --release -- --weights-url https://github.com/vous/ecriture-rs/releases/download/v1.0
# ou pointer vers un dossier local :
cargo run --release -- --weights-dir ./pylaia-rimes

# modèle IAM Sachin local :
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
│   ├── ink.rs          Point/Stroke, resample, lissage, rendu image grayscale avec marges, export InkML
│   ├── lexicon.rs      lexique de langue + couverture lexicale d'une hypothèse
│   ├── tablet.rs       lecture evdev du stylet (thread + channel + réveil egui)
│   ├── model.rs        CRNN "Sachin" candle-nn, chargement .pth (pthrs), téléchargement HF
│   ├── doctr.rs        CRNN doctR mobilenet_v3_large (pytorch_model.bin)
│   ├── pylaia.rs       CRNN PyLaIA LaiaCRNN (weights.ckpt + syms.txt, ex. RIMES FR)
│   └── recognizer.rs   trait Recognizer, 3 backends, beam CTC N-best + rescoring lexique
│   └── bin/
│       └── uinput-pilot.rs  pilote de périphériques virtuels (souris/stylet) via /dev/uinput
├── tools/
│   └── scenario-stylet-puis-souris.txt  scénario d'injection uinput-pilot (exemple)
├── examples/
│   ├── dump_pth.rs     liste les tenseurs d'un checkpoint torch
│   ├── export_pth.rs   exporte un checkpoint en .f32 bruts (debug cross-framework)
│   └── to_safetensors.rs  convertit un .ckpt torch en .safetensors (f32/f16) pour partager les poids
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
lexique** (`--lm-weight`, lexique français embarqué par défaut). Benchmarks réels
(tests `#[ignore]`) :
- IAM (Sachin, 10 lignes, lexique anglais de calibration) : CER glouton 0.1025 →
  beam+lexique **0.0968** (modèle annoncé à 0.126 sur tout son test)
- RIMES (PyLaIA FR, 12 lignes test) : CER beam ≈ **0.29** sur ces 12 premières
  lignes (le modèle annonce 4.5 % sur les 778 lignes du test)
- Parité Rust vs PyTorch validée sur les 3 backends (logits max|diff| < 6e-5) :
  `cargo test --release -- --ignored parite`

## Tester l'input (record/replay + pilote uinput)

Deux outils 100 % libres pour reproduire et diagnostiquer les bugs d'interface
(souris qui trace en hover, stylet bloqué, etc.) sans cliquer à la main :

**Record / replay déterministe** — enregistrer une session d'input et la rejouer
en headless (sans fenêtre, modèle simulé), ex. en CI :

```bash
# enregistrer tous les événements (souris, touch, tablette evdev) en JSONL :
cargo run --release -- --record-input /tmp/events.jsonl
# ... reproduire le bug à la souris/stylet, quitter l'app ...
# rejouer le fichier : affiche les strokes produits frame par frame :
cargo run --release -- --replay-input /tmp/events.jsonl
```

**Pilote uinput** — injecter de vrais périphériques virtuels (souris + stylet)
dans la session via `/dev/uinput` (groupe `input`, comme pour la lecture evdev),
pilotés par un fichier scénario :

```bash
cargo run --release --bin uinput-pilot -- tools/scenario-stylet-puis-souris.txt
# commandes : screen W H | mouse | pen | mmove dx dy | mdown/mup/mclick |
#             pmove x y | pdown [p] | pup | sleep ms   (# = commentaire)
```

Combiner les deux : lancer l'app avec `--record-input`, injecter un scénario
avec `uinput-pilot`, puis rejouer le JSONL — le bug est figé et rejouable à
l'identique (couvert par les tests `record_puis_replay_*`).

## Roadmap

- **Modèle de trajectoires (online)** : `ink::strokes_to_features` produit déjà les features `[1, T, 8]` (x, y, dx, dy, dt, pression, cos/sin) ; `Recognizer::recognize_strokes` existe (fallback image CRNN pour l'instant). **Export InkML fait** (bouton *Exporter InkML* → `exports/ink-<date>.inkml`, channels X/Y/T/P + vérité terrain) pour constituer sa propre base de tracés online (compléter par IAM-OnDB / IRONOFF / MathWriting-InkML). Reste à brancher un LRU/petit transformer exporté en ONNX (`[1, T, F]`), inféré en pur Rust via **rten** (ou candle), avec état récurrent entre les appels. Le CRNN image reste la baseline mesurable (latence + CER).
- ~~**Fenêtre glissante** avec hypothèses révisables~~ fait : les strokes non finalisés sont re-reconnus au debounce (0,6 s) et remplacés (hypothèse en gris) ; figés après 2 s de pause ou au-delà de 12 strokes.
- **Segmentation** en mots/gestes (double-tap = espace, geste = retour à la ligne) via un petit classifieur séparé.
- Quantifier le modèle (int8) et viser < 50 ms par fenêtre ; valider le CER sur tes propres strokes.

## Licence

[GPL v2](LICENSE.md)
