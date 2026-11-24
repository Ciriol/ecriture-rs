# ecriture-rs

Application Rust de prise de notes avec reconnaissance d'écriture quasi temps réel.
Pipeline : capture de strokes (egui) → inférence ONNX locale (candle) → texte.

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

### 4. Brancher un vrai modèle
Le `MockRecognizer` sert à valider l'UI ; il reste à brancher le vrai modèle :

- Placer `model.onnx` et `chars.txt` (un caractère par ligne, dans l'ordre des classes du modèle) à la racine du projet.
- Configurable en CLI : `cargo run --release -- --model model.onnx --chars chars.txt --input-h 64 --norm 01|neg1` (`--input-h` = hauteur d'entrée du modèle, `--norm 01` = pixels 0..1 blanc=1, `neg1` = -1..1).
- Vérifier que `CandleRecognizer` charge le graphe (noms d'entrée/sortie lus depuis le proto ONNX) et que le décodage CTC produit du texte cohérent sur un échantillon connu.
- Comparer les sorties modèle réel vs mock sur les mêmes strokes.
