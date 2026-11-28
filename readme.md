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
fenêtre = un seul glyphe compact ? (digits::looks_like_single_glyph)
        │oui
        ▼
LR softmax MNIST 28x28 (src/digits.rs, poids ~31 Ko embarqués
assets/digits-lr.f32) → chiffre 0-9 si confiance >= --digits-threshold
        │non (ou confiance insuffisante)
        ▼
rendu en image grayscale avec marges + lissage (src/ink.rs)
        │
        ▼
CRNN CNN-BiLSTM-CTC en candle, poids chargés depuis un .pth (src/model.rs, pthrs)
        │
        ▼
beam search CTC N-best en log-space (src/recognizer.rs) avec shallow fusion d'un
LM de caractères uni/bigrammes pendant le beam (src/charlm.rs, construit depuis
le lexique — inspiré de githubharald/CTCDecoder) + rescoring des N-best par
lexique de langue (src/lexicon.rs, ~49k mots embarqués dans assets/) → texte affiché
```

Le classifieur de chiffres isolés est une régression logistique multinomiale
(softmax 784→10) entraînée sur MNIST, en pur Rust et embarquée — inspiré de
[sktan888/handwriting_recognition_logistic_regression](https://github.com/sktan888/handwriting_recognition_logistic_regression)
(voir *Crédits*). Il fiabilise la détection des chiffres isolés (dates,
montants, numéros) que le CRNN peut confondre (1/l, 0/O, 5/S) : accuracy
~92 % sur le test MNIST complet, ~95 % sur les 100 échantillons embarqués
(`assets/mnist-samples.bin`, test unitaire). Réentraînement hors-ligne :
`cargo run --release --bin train-digits-lr -- <dossier MNIST IDX>`. Le backend
**chiffres isolés** du menu déroulant reconnaît un glyphe unique avec ce LR.

- Reconnaissance **incrémentale** : 0,6 s après la fin d'un tracé, seuls les nouveaux strokes sont reconnus et ajoutés au texte.
- **Stylet Linux** : lu directement via evdev (`src/tablet.rs`) quand winit ne remonte rien ; contact détecté par `BTN_TOUCH`/pression, coordonnées normalisées depuis les bornes du périphérique, pression `ABS_PRESSURE` conservée dans chaque point. Les coordonnées normalisées sont projetées sur le moniteur puis rapportées à la position réelle de la fenêtre (`viewport.inner_rect`) : en fenêtré, seule la zone physique de l'écran couverte par la fenêtre est dessinable (plein écran : identique à avant).
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
| `--renderer glow\|wgpu` | `wgpu` (Vulkan) | backend de rendu eframe. Sur les vieux iGPU Intel (Haswell), Mesa Vulkan est incomplet (`MESA-INTEL: warning: Haswell Vulkan support is incomplete`, risques de crash) : utiliser `--renderer glow` (OpenGL), ou env `ECRITURE_RENDERER=glow` |
| `--word-seg` | off | segmente la ligne en mots (`src/wordseg.rs`, méthode scale-space de Manmatha & Srimal, inspirée de githubharald/WordDetector : noyau gaussien anisotrope + Otsu + composantes connexes, 100 % local sans poids) puis reconnaît **chaque mot séparément**, joints par des espaces explicites ; repli sur la ligne entière si un seul mot est détecté |
| `--lexicon PATH\|none` | lexique français embarqué (`assets/lexique-fr.txt`) | fichier "1 mot par ligne" pour le rescoring N-best ; `none` désactive le lexique |
| `--beam N` | `16` | largeur du beam search CTC (`1` = décodage glouton) |
| `--nbest N` | `8` | nombre d'hypothèses N-best rescorées par le lexique |
| `--lm-weight F` | `2.0` | poids du lexique : `score = logp_ctc/len + F × couverture_lexicale` (`0` = désactivé) |
| `--char-lm-weight F` | `0.4` | poids du LM de caractères (unigrammes+bigrammes, construit depuis le lexique actif) en **shallow fusion pendant le beam CTC** — inspiré de [githubharald/CTCDecoder](https://github.com/githubharald/CTCDecoder) (`beam_search` + `LanguageModel`) : chaque extension de préfixe ajoute `F × ln P(c \| dernier char)`, score LM normalisé par la longueur en fin de beam. `0` = désactivé ; sans lexique (`--lexicon none`) pas de LM char |
| `--digits auto\|off\|PATH` | `auto` | classifieur de chiffres isolés (LR softmax MNIST) : `auto` = poids embarqués `assets/digits-lr.f32`, `off` = désactivé (que le CRNN), `PATH` = poids custom (.f32 brut 31400 o ou .json `{"w":[[…784…]x10],"b":[…10…]}`, repli sur l'embarqué si illisible) |
| `--digits-threshold F` | `0.85` | confiance softmax minimale pour qu'un glyphe isolé soit reconnu comme chiffre 0-9 par le LR ; sinon repli sur le CRNN |

Dans l'UI : tracer dans la zone centrale, **Reconnaître** (tout relire), **Effacer**, **Exporter InkML** (sauve les tracés dans `exports/ink-<date>.inkml`, format W3C InkML : un `<trace>` par stroke, channels X/Y/T/P + `<annotation type="truth">` avec le label), case **debug** (événements d'entrée reçus). Le champ **vérité terrain** (au-dessus du texte reconnu) est pré-rempli avec le texte reconnu et reste éditable : corrigez-le avant d'exporter pour constituer une base de tracés online (entraînement d'un modèle de trajectoires). En haut à droite, le **menu déroulant indique le modèle actif** (source, backend, taille d'entrée) et permet d'en **changer à chaud** — le chargement (téléchargement HF + parsing du checkpoint) se fait en arrière-plan, l'UI affiche « chargement… » puis re-reconnaît le texte à l'écran avec le nouveau modèle. Choix proposés : `Sachin-0001/crnn-iam-handwriting` (IAM anglais), `Teklia/pylaia-rimes` (PyLaIA français), `Felix92/doctr-torch-crnn-mobilenet-v3-large-french` (doctR français), **chiffres isolés 0-9 · LR softmax MNIST** (poids embarqués, reconnaît un glyphe unique), plus la source passée en CLI si c'est un modèle local. Hors menu, chaque modèle CRNN est automatiquement enveloppé dans un `HybridRecognizer` : un glyphe isolé dont la confiance LR >= `--digits-threshold` est reconnu comme chiffre, le reste part au CRNN. Le panneau du **texte reconnu en bas est redimensionnable** (tirer sa bordure haute, 120–500 px, 200 px par défaut) : texte finalisé en 28 pt, hypothèse de la fenêtre glissante en gris en dessous.

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
│   ├── lexique-fr.txt      lexique français embarqué (~49k mots, fréquences corpus)
│   ├── digits-lr.f32       poids LR softmax MNIST embarqués (10x784 + 10 f32 LE, 31400 o)
│   └── mnist-samples.bin   100 échantillons test MNIST embarqués (u32 count + label/784 px)
├── src/
│   ├── lib.rs          lib ecriture_rs (modules partagés bin + outils)
│   ├── main.rs         UI egui, capture souris/touch, debounce, options CLI
│   ├── digits.rs       classifieur de chiffres isolés : LR softmax 784->10 (poids embarqués
│   │                   ou .f32/.json custom), rendu strokes 28x28, heuristique glyphe isolé,
│   │                   lecteur IDX MNIST + entraînement SGD (entraînement hors-ligne)
│   ├── ink.rs          Point/Stroke, resample, lissage, rendu image grayscale avec marges, export InkML
│   ├── charlm.rs       LM de caractères (uni/bigrammes, backoff, pondération Zipf depuis le
│   │                   lexique) pour la shallow fusion dans le beam CTC (style CTCDecoder)
│   ├── lexicon.rs      lexique de langue + couverture lexicale d'une hypothèse
│   ├── tablet.rs       lecture evdev du stylet (thread + channel + réveil egui)
│   ├── model.rs        CRNN "Sachin" candle-nn, chargement .pth (pthrs), téléchargement HF
│   ├── doctr.rs        CRNN doctR mobilenet_v3_large (pytorch_model.bin)
│   ├── pylaia.rs       CRNN PyLaIA LaiaCRNN (weights.ckpt + syms.txt, ex. RIMES FR)
│   ├── recognizer.rs   trait Recognizer, backends CRNN + DigitRecognizer + HybridRecognizer
│   │                   (glyphe isolé -> LR chiffres si confiance >= seuil, sinon CRNN),
│   │                   beam CTC N-best + rescoring lexique
│   └── bin/
│       ├── train-digits-lr.rs  entraîne le LR MNIST depuis un dossier IDX (60k images,
│       │                       ~16 s en release, accuracy ~92 %) et écrit assets/digits-lr.f32
│       │                       + assets/mnist-samples.bin
│       └── uinput-pilot.rs  pilote de périphériques virtuels (souris/stylet) via /dev/uinput
├── tools/
│   └── scenario-stylet-puis-souris.txt  scénario d'injection uinput-pilot (exemple)
├── examples/
│   ├── dump_pth.rs     liste les tenseurs d'un checkpoint torch
│   ├── export_pth.rs   exporte un checkpoint en .f32 bruts (debug cross-framework)
│   └── to_safetensors.rs  convertit un .ckpt torch en .safetensors (f32/f16) pour partager les poids
└── LICENSE.md
```

## Organisation multi-machines (dev ↔ yogarch)

Deux machines, deux rôles, sessions archivées ensemble dans l'arbre `vibe` :

| Machine | Rôle | Sessions |
|---|---|---|
| poste de dev (32 Go RAM) | modèles de reconnaissance, conversions de poids, entraînements, tests de parité gourmands (`cargo test --release -- --ignored`) | branche `vibe-reconnaissance` |
| **yogarch** (`192.168.1.34`, plateforme cible, ~8 Go RAM) | interface (fenêtre, tablette evdev, souris/hover), tests directs sur la cible via `uinput-pilot` + record/replay | branche `vibe-interface` |

- Le hook `session_end` archive chaque session dans sa branche par catégorie
  (mots-clés du titre) ; `git fetch vibe-archive` ramène les sessions de
  l'autre machine, le rebuild les fusionne par date dans `vibe` /
  `vibe-interface` / `vibe-reconnaissance` — poussées vers le dépôt **privé**
  `Ciriol/vibe-sessions` (remote `vibe-archive`, `git config
  vibe.archiveRemote`), jamais vers le dépôt public `origin`.
- SSH vers yogarch : alias `yogarch` (`~/.ssh/config`) + clé
  `~/.ssh/id_perso_yogarch` (ed25519, générée sur yogarch puis privée
  rapatriée ici, 600 ; la privée a été supprimée de yogarch). Repli mot de
  passe via askpass lisant `~/.vibe/archive-scrub-local.txt` (chmod 600,
  hors dépôt, masqué dans l'archive vibe) — voir `archive-vibe-sessions`.
- Pour lancer une tâche lourde sur yogarch : `ssh ciriol@192.168.1.34 'cd
  ~/Projets/ecriture-rs && cargo test --release -- --ignored bench_iam'`.
- Pilotage sans confirmation : wrappers `yog '<cmd>'` et `yog-scp <src> <dst>`
  dans `~/.local/bin` (clé `id_perso_yogarch`, repli askpass), préfixes
  auto-approuvés dans `~/.vibe/config.toml` `[tools.bash] allowlist`
  (`yog`, `yog-scp`, `ssh yogarch`, `scp yogarch:`) — s'applique après
  `/reload` ou à la prochaine session.

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
- Classifieur de chiffres isolés (LR softmax MNIST embarqué) : accuracy
  **0.95** sur les 100 échantillons MNIST embarqués (0.9237 sur les 10 000 du
  test complet, entraînement `train-digits-lr`) — test unitaire
  `classifieur_embarque_reconnait_les_echantillons_mnist` ; le routage
  glyphe-isolé→LR vs mot→CRNN est couvert par `hybride_glyphe_isole_*`.
- Beam CTC log-space + LM de caractères : parité avec l'ancien beam sans LM
  (`ctc_beam_decode_meme_regles_que_greedy`, `ctc_beam_prefere_...`), stabilité
  `logaddexp`, shallow fusion LM qui fait gagner le bigram vu
  (`beam_lm_char_fait_pencher_vers_le_bigram_vu`), LM embarqué FR cohérent
  (`lm_embarque_francais_est_coherent` : P(e|l) > P(z|l), P(u|q) > P(a|q)).

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
# ou mode interactif (lit les commandes sur stdin, devices persistants) :
cargo run --release --bin uinput-pilot -- -
# commandes : screen W H | mouse | amouse | pen |
#   mmove dx dy | mdown/mup/mclick            (souris relative)
#   amove x y | adown/aup                     (souris ABSOLUE : positionne le
#                                              curseur exactement, utile sous
#                                              Wayland où wl_pointer.motion n'est
#                                              envoyé que si le curseur est sur
#                                              la fenêtre)
#   pmove x y | pdown [p] | pup               (stylet : contact BTN_TOUCH)
#   phover x y [p] | pleave                   (proximité stylet, pression sans
#                                              BTN_TOUCH : ne doit pas tracer)
#   sleep ms                                  (# = commentaire)
```

Combiner les deux : lancer l'app avec `--record-input`, injecter un scénario
avec `uinput-pilot`, puis rejouer le JSONL — le bug est figé et rejouable à
l'identique (couvert par les tests `record_puis_replay_*`).

## Archiver les sessions Vibe (arbre `vibe`, dépôt privé, sans code)

Le hook `session_end` (`.vibe/hooks.toml`, skill
`.agents/skills/archive-vibe-sessions`) archive les sessions journalisées
localement (`~/.vibe/logs/session/unified/<id>/`) dans l'arbre `vibe` — tronc
général `vibe` + branches `vibe-interface` / `vibe-reconnaissance`, un commit
par session, ordonné par `start_time`, secrets masqués — puis pousse
(`--force-with-lease`) vers le dépôt **privé** `Ciriol/vibe-sessions` via le
remote `vibe-archive`. **Aucun fichier du projet n'y est poussé** (vérifié :
`git ls-tree -r <branche> | grep -v '^sessions/'` → vide). Chaque session :
`sessions/<slug>/{RESUME.md, journal.jsonl, meta.json, records/}` + index
global `sessions/{README.md,index.json}`. Le worktree courant n'est jamais
modifié (plumbing git sur index temporaire). Export manuel vers un dossier
temporaire (jamais `--branch`, réservé au passé) :

```bash
cargo run --bin session-export -- --list                      # sessions trouvées
cargo run --bin session-export -- --export <id> --out /tmp/vibe-sessions-export
python3 .agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py /tmp/vibe-sessions-export
git push --force-with-lease vibe-archive vibe vibe-interface vibe-reconnaissance
```

## Roadmap

- **Modèle de trajectoires (online)** : `ink::strokes_to_features` produit déjà les features `[1, T, 8]` (x, y, dx, dy, dt, pression, cos/sin) ; `Recognizer::recognize_strokes` existe (fallback image CRNN pour l'instant). **Export InkML fait** (bouton *Exporter InkML* → `exports/ink-<date>.inkml`, channels X/Y/T/P + vérité terrain) pour constituer sa propre base de tracés online (compléter par IAM-OnDB / IRONOFF / MathWriting-InkML). Reste à brancher un LRU/petit transformer exporté en ONNX (`[1, T, F]`), inféré en pur Rust via **rten** (ou candle), avec état récurrent entre les appels. Le CRNN image reste la baseline mesurable (latence + CER).
- ~~**Fenêtre glissante** avec hypothèses révisables~~ fait : les strokes non finalisés sont re-reconnus au debounce (0,6 s) et remplacés (hypothèse en gris) ; figés après 2 s de pause ou au-delà de 12 strokes.
- **Segmentation** en mots/gestes (double-tap = espace, geste = retour à la ligne) via un petit classifieur séparé. Brique déjà en place : le classifieur de glyphe isolé (`src/digits.rs`, LR MNIST) + l'heuristique `looks_like_single_glyph` routent déjà les glyphes uniques vers un classifieur dédié dans `HybridRecognizer`.
- Quantifier le modèle (int8) et viser < 50 ms par fenêtre ; valider le CER sur tes propres strokes.

## Crédits

- Le classifieur de chiffres isolés (`src/digits.rs`, poids `assets/digits-lr.f32`)
  s'inspire de
  [sktan888/handwriting_recognition_logistic_regression](https://github.com/sktan888/handwriting_recognition_logistic_regression)
  : régression logistique (softmax) sur MNIST 28x28, 10 classes. Ici réimplémenté
  en pur Rust embarqué (entraînement `src/bin/train-digits-lr.rs` sur les fichiers
  IDX MNIST du dépôt upstream, inférence = matmul 784x10 + softmax, aucun moteur ML).
- Le **LM de caractères en shallow fusion pendant le beam CTC** (`src/charlm.rs`
  + `ctc_beam_decode_nbest_lm` dans `src/recognizer.rs`, beam réécrit en
  log-space) reprend l'algorithme de
  [githubharald/CTCDecoder](https://github.com/githubharald/CTCDecoder)
  (`beam_search` + `apply_lm` + `BeamList.normalize`), réimplémenté en pur Rust :
  le dépôt upstream est en Python, incompatible avec la cible 100 % Rust du
  projet. Son `lexicon_search`/BK-tree est déjà couvert côté Rust par le
  beam N-best + rescoring par lexique de mots.

## Licence

[GPL v2](LICENSE.md)
