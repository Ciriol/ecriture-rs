# ecriture-rs

Application Rust de prise de notes avec reconnaissance d'écriture quasi temps réel, 100 % Rust.
Pipeline : capture de strokes (egui souris/touch, evdev pour le stylet) → inférence locale (candle, CRNN depuis un checkpoint .pth) → texte CTC.

## À faire

### 1. Passer à un modèle stroke-based (vrai temps réel)
Aujourd'hui, les strokes sont rendus en image (`ink::render_to_image`) puis reconnus par un modèle image (CRNN/CTC). Passer directement sur les séquences de points (x, y, t, pression) :

- Choisir ou exporter un modèle ONNX de séquences, entrée attendue `[1, T, features]` (features = x, y, t, pression normalisés).
- Normaliser les strokes en entrée : rééchantillonner à `T` points fixes (s'appuyer sur `Stroke::resample`), centrer et mettre à l'échelle via `bounding_box`.
- Ajouter une méthode `recognize_strokes(&[Stroke])` au trait `Recognizer` dans `recognizer.rs`, en parallèle de `recognize(&GrayImage)`.
- Adapter le décodage (CTC glouton ou seq2seq glouton) au format de sortie du modèle choisi.
- Mesurer la latence par stroke pour valider le « quasi temps réel ».

### 2. Fenêtre glissante avec hypothèses révisables
Le texte reconnu doit pouvoir se corriger en direct au lieu de s'accumuler figé :

- Garder une fenêtre des `K` derniers strokes ; à chaque debounce (`DEBOUNCE_SECS` dans `main.rs`), relancer la reconnaissance sur toute la fenêtre et remplacer la portion de texte correspondante dans `result`.
- Suivre quels strokes sont « finalisés » vs « en cours de révision » (aujourd'hui `recognized_up_to` fige tout ce qui a été reconnu).
- Afficher différemment dans l'UI le texte finalisé et le texte encore révisable.
- Définir les règles de finalisation : nouveau mot, pause prolongée, geste de séparation.

### 3. Segmentation en mots et gestes
Ajouter des gestes de contrôle au-delà du simple tracé :

- Double-tap : insérer un espace (nouveau mot).
- Geste dédié (ex. trait horizontal de droite à gauche) : retour à la ligne.
- Pause longue sans tracé : séparateur implicite entre deux unités de texte.
- Refléter espaces et retours à la ligne dans le panneau de résultat.

### 4. Modèle réel (déjà branché en natif Rust)
Le CRNN (CNN-BiLSTM-CTC) est chargé directement depuis un checkpoint PyTorch `.pth` : lecture pure Rust via `pthrs`, poids téléchargés depuis Hugging Face via `hf-hub`, inférence candle. Aucun Python ni ONNX.

- Par défaut : `cargo run --release` télécharge `Sachin-0001/crnn-iam-handwriting` (model.pth, vocab.json, config.json) — hauteur d'entrée 32, normalisation [-1, 1] prises depuis `config.json`.
- Modèle local : `--pth /chemin/model.pth` (vocab.json et config.json attendus à côté).
- Options CLI : `--repo owner/name`, `--input-h N`, `--norm 01|neg1`, `--tablet-device /dev/input/eventN`.
- Reste à faire : comparer les sorties modèle réel vs `MockRecognizer` sur les mêmes strokes, et valider la qualité sur de l'écriture réelle.
