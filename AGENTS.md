# AGENTS.md — ecriture-rs

Prise de notes avec reconnaissance d'écriture manuscrite quasi temps réel :
**100 % Rust, inférence 100 % locale** (candle, CPU) — aucun Python, aucun ONNX,
aucun cloud au runtime. Licence **GPL-2.0**. Readme complet (fonctionnement,
architecture, utilisation) : `readme.md`. Docs, UI et commits en **français**.

## Prérequis

- Rust 1.99+ (rustup) ; Linux pour le support stylet.
- Stylet : droits de lecture sur `/dev/input` (`sudo usermod -aG input $USER`, puis re-login).

## Commandes

```bash
cargo build                                        # build
cargo run --release                                # lancer l'app (options : --pth, --repo, --tablet-device, --record-input… voir readme "Lancer")
cargo test                                         # tests unitaires (rendu, CTC greedy/beam, vocab, LR chiffres)
cargo test --release -- --ignored --nocapture      # test réel : télécharge le modèle HF et infère
cargo test --release -- --ignored parite           # parité Rust vs PyTorch (3 backends)
cargo test --release -- --ignored bench_iam        # bench CER sur vraies lignes IAM
cargo run --release --bin uinput-pilot -- ...      # pilote /dev/uinput (record/replay de scénarios d'input)
cargo run --bin session-export -- --list          # sessions Vibe journalisées (voir "Sessions Vibe")
```

## Plan du dépôt

- `src/main.rs` : binaire de l'app (egui/eframe).
- `src/lib.rs` + modules — pipeline de reconnaissance :
  `ink.rs` (strokes → image), `model.rs` / `pylaia.rs` / `doctr.rs` (backends CRNN-CTC, poids `.pth` chargés via `pthrs`),
  `recognizer.rs` (beam search CTC N-best + rescoring), `charlm.rs` + `lexicon.rs` (LM de caractères + lexique FR embarqué),
  `digits.rs` (classifieur de chiffres isolés, LR softmax MNIST embarquée), `tablet.rs` (stylet evdev),
  `wordseg.rs` (segmentation de la ligne en mots, scale-space inspiré de githubharald/WordDetector, mode `--word-seg`).
- `src/bin/` : `session-export` (export des sessions Vibe), `train-digits-lr` (réentraînement offline du LR MNIST), `uinput-pilot` (record/replay input).
- `assets/` : poids et lexique embarqués (`digits-lr.f32`, `lexique-fr.txt`, `mnist-samples.bin`).
- `tools/export-model/` : conversions de poids **hors ligne** (Python autorisé ici uniquement, jamais au runtime).
- `examples/` : `to_safetensors`, `export_pth`, `dump_pth` (partage de poids).

## Contraintes dures

- **Stylet** : evdev Linux uniquement ; seuls les périphériques stylet (`BTN_TOOL_PEN`) sont lus — ne pas « corriger » ce filtre.
- **Parité bit-à-bit vs PyTorch** requise sur les 3 backends : logits `max|diff| < 6e-5` (tests `parite_*`).
- **Prétraitement** (contrat) : bounding box à 75 % de la hauteur, largeur min 2×hauteur, trait ~hauteur/10, lissage du bruit, resize **bilinéaire** vers `--input-h` (32), normalisation `p/127.5 - 1`.
- Poids des modèles : cache HF (`--repo`) ou fichiers locaux (`--pth` / `--weights-dir`) ; l'inférence reste locale.
- Ne pas introduire de dépendance Python/ONNX/cloud au runtime.

## Sessions Vibe (OBLIGATOIRE)

- Charger et suivre le skill `.agents/skills/archive-vibe-sessions/SKILL.md` : toute session Vibe de ce projet est archivée dans l'arbre `vibe` (tronc général + branches `vibe-interface` / `vibe-reconnaissance`, un commit par session, ordonné par `start_time`, dates = `start_time`).
- Push **uniquement** vers le remote `vibe-archive` (dépôt **privé** `Ciriol/vibe-sessions`, `git config vibe.archiveRemote`) — **jamais** vers `origin` (dépôt public) : `git push --force-with-lease vibe-archive vibe vibe-interface vibe-reconnaissance`.
- Jamais de branche `vibe/...` ni `vibe/session-*` ; l'export passe par un dossier temporaire (`session-export --out <dir>`).
- Secrets masqués (`***REDACTED***` ; `rebase_vibe_sessions.py --scan` doit trouver 0 secret) ; **aucun fichier du projet** dans les branches `vibe*` (vérifié : `git ls-tree -r <branche> | grep -v '^sessions/'` → vide).
- Le hook `session_end` (`.vibe/hooks.toml`) effectue ce travail automatiquement à la fermeture de la session ; avant de terminer une session, vérifier qu'il a réussi, sinon lancer `python3 .agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py` puis le push.
- Organisation multi-machines (dev ↔ yogarch) et détails : `readme.md` et le README de `Ciriol/vibe-sessions`.
