//! UI egui : canvas d'écriture + bouton de reconnaissance.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};

use anyhow::Context as _;
use eframe::egui;
use ecriture_rs::digits::DigitClassifier;
use ecriture_rs::ink::{self, Point, Stroke};
use ecriture_rs::recognizer::{
    CrnnRecognizer, DecodeOptions, DigitRecognizer, HybridRecognizer, MockRecognizer, Norm,
    Recognizer,
};
use ecriture_rs::tablet::{self, TabletEvent};
use ecriture_rs::{doctr, lexicon, model};

/// Délai de pause (en secondes) après lequel on re-reconnaît la fenêtre en cours.
const DEBOUNCE_SECS: f64 = 0.6;
/// Pause plus longue (s) après laquelle la fenêtre en cours est figée dans le texte.
const FINALIZE_PAUSE_SECS: f64 = 2.0;
/// Nombre max de strokes non finalisés dans la fenêtre glissante.
const WINDOW_MAX_STROKES: usize = 12;

struct InkApp {
    strokes: Vec<Stroke>,
    current: Option<Stroke>,
    /// Texte finalisé (figé).
    result: String,
    /// Hypothèse courante de la fenêtre glissante (révisable, affichée en gris).
    pending_text: String,
    /// Nombre de strokes finalisés (intégrés dans `result`).
    finalized: usize,
    /// Temps (s) du dernier point tracé.
    last_draw_time: f64,
    recognizer: Box<dyn Recognizer>,
    /// Un touch (stylet/doigt) est actuellement actif.
    touch_active: bool,
    /// Bouton principal de la souris réellement enfoncé. État suivi nous-mêmes
    /// à partir des événements bruts : `pointer.any_down()` peut rester bloqué
    /// après un contact stylet sous Wayland (le release du PointerButton émulé
    /// n'arrive jamais), ce qui faisait tracer la souris en hover.
    mouse_down: bool,
    /// Temps du dernier événement Touch reçu (détecte un touch bloqué : Start sans End).
    last_touch_event: f64,
    /// Temps du dernier événement tablette evdev reçu (détecte un contact bloqué : Up manqué).
    last_tablet_event: f64,
    /// Un contact stylet est actif via evdev (aucun événement winit reçu).
    tablet_active: bool,
    /// Dernière position connue d'un touch (si le backend n'émule pas la souris).
    last_touch_pos: Option<egui::Pos2>,
    /// Mode debug : affiche les événements d'entrée reçus.
    debug: bool,
    debug_info: String,
    /// Image envoyée au modèle lors de la dernière reconnaissance (mode debug).
    debug_img: Option<egui::ColorImage>,
    /// Réception des événements du stylet lus via evdev.
    tablet_rx: Option<Receiver<TabletEvent>>,
    /// Compteur d'événements tablette reçus (pour le mode debug).
    tablet_count: u64,
    /// Sources de modèles proposées dans le menu déroulant (haut à droite).
    model_sources: Vec<ModelSource>,
    /// Index dans `model_sources` du modèle sélectionné (actif ou en chargement).
    model_index: usize,
    /// Description du modèle actif (backend, taille d'entrée, erreur...).
    model_desc: String,
    /// Un chargement de modèle est en cours en arrière-plan.
    model_loading: bool,
    /// Réception du reconnaisseur chargé en arrière-plan (bascule à chaud).
    model_rx: Option<Receiver<(Box<dyn Recognizer>, String)>>,
    /// Config CLI (pour recharger un modèle avec les mêmes options).
    cfg: Config,
    /// Options de décodage (beam, lexique...) appliquées à chaque modèle.
    decode_opts: DecodeOptions,
    /// Label de vérité terrain pour l'export InkML (pré-rempli avec le texte
    /// reconnu, éditable manuellement avant export).
    export_label: String,
    /// Dernier label auto-généré (pour ne pas écraser une correction manuelle).
    last_auto_label: String,
    /// Statut du dernier export InkML (chemin écrit ou erreur).
    export_status: String,
    /// Enregistreur d'input (--record-input) : une ligne JSONL par frame egui
    /// et par batch d'événements tablette evdev.
    recorder: Option<std::io::BufWriter<std::fs::File>>,
    /// Nombre de frames enregistrées (flush périodique du BufWriter).
    recorded_frames: u64,
}

/// Une ligne d'un fichier d'enregistrement d'input (JSONL) : soit une frame
/// egui complète (`RawInput` sérialisé, avec time + events), soit un batch
/// d'événements tablette lus via evdev (qui ne passent pas par winit).
#[derive(serde::Serialize, serde::Deserialize)]
enum RecordedLine {
    Frame(egui::RawInput),
    Tablet(Vec<TabletEvent>),
}

impl InkApp {
    fn new(
        recognizer: Box<dyn Recognizer>,
        model_desc: String,
        model_sources: Vec<ModelSource>,
        model_index: usize,
        cfg: Config,
        decode_opts: DecodeOptions,
        tablet_device: Option<String>,
        ctx: egui::Context,
    ) -> Self {
        // Lance la lecture evdev des périphériques stylet/touch en arrière-plan.
        // Le thread evdev réveille egui (request_repaint) à chaque événement.
        let (tx, rx) = channel();
        tablet::spawn_readers(tx, tablet_device, ctx);
        let mut app = Self::with_tablet_rx(
            recognizer,
            model_desc,
            model_sources,
            model_index,
            cfg.clone(),
            decode_opts,
            Some(rx),
        );
        // --record-input : enregistre chaque frame dans un fichier JSONL.
        app.recorder = cfg.record_input.as_ref().map(|p| {
            let f = std::fs::File::create(p).unwrap_or_else(|e| {
                eprintln!("record-input: impossible de créer {p}: {e}");
                std::process::exit(1);
            });
            eprintln!("record-input: écriture des événements dans {p}");
            std::io::BufWriter::new(f)
        });
        app
    }

    /// Construit l'app avec le channel tablette fourni (`None` = pas de
    /// lecture evdev ; utilisé par les tests).
    fn with_tablet_rx(
        recognizer: Box<dyn Recognizer>,
        model_desc: String,
        model_sources: Vec<ModelSource>,
        model_index: usize,
        cfg: Config,
        decode_opts: DecodeOptions,
        tablet_rx: Option<Receiver<TabletEvent>>,
    ) -> Self {
        Self {
            strokes: Vec::new(),
            current: None,
            result: String::new(),
            pending_text: String::new(),
            finalized: 0,
            last_draw_time: 0.0,
            recognizer,
            touch_active: false,
            mouse_down: false,
            last_touch_event: 0.0,
            last_tablet_event: 0.0,
            tablet_active: false,
            last_touch_pos: None,
            debug: false,
            debug_info: String::new(),
            debug_img: None,
            tablet_rx,
            tablet_count: 0,
            model_sources,
            model_index,
            model_desc,
            model_loading: false,
            model_rx: None,
            cfg,
            decode_opts,
            export_label: String::new(),
            last_auto_label: String::new(),
            export_status: String::new(),
            recorder: None,
            recorded_frames: 0,
        }
    }

    /// Enregistre la frame egui courante (RawInput complet : time, screen_rect,
    /// events...) comme une ligne JSONL. Flush à chaque ligne : un kill brutal
    /// de l'app ne doit rien perdre.
    fn record_frame(&mut self, ctx: &egui::Context) {
        use std::io::Write as _;
        if let Some(w) = &mut self.recorder {
            let raw = ctx.input(|i| i.raw.clone());
            if let Ok(line) = serde_json::to_string(&RecordedLine::Frame(raw)) {
                let _ = writeln!(w, "{line}");
                let _ = w.flush();
                self.recorded_frames += 1;
            }
        }
    }

    /// Enregistre un batch d'événements tablette (evdev) reçus cette frame.
    fn record_tablet(&mut self, events: &[TabletEvent]) {
        use std::io::Write as _;
        if let Some(w) = &mut self.recorder {
            if let Ok(line) = serde_json::to_string(&RecordedLine::Tablet(events.to_vec())) {
                let _ = writeln!(w, "{line}");
                let _ = w.flush();
            }
        }
    }

    /// Flush final du fichier d'enregistrement.
    fn flush_recorder(&mut self) {
        use std::io::Write as _;
        if let Some(w) = &mut self.recorder {
            let _ = w.flush();
        }
    }

    /// Met à jour le label d'export (vérité terrain) avec le texte reconnu,
    /// sans écraser une correction manuelle de l'utilisateur.
    fn sync_export_label(&mut self) {
        let mut cur = self.result.clone();
        if !self.pending_text.is_empty() {
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(&self.pending_text);
        }
        if self.export_label.is_empty() || self.export_label == self.last_auto_label {
            self.export_label = cur.clone();
            self.last_auto_label = cur;
        }
    }

    /// Exporte les strokes courants en InkML dans `exports/` (relatif au CWD).
    fn do_export(&mut self) {
        if self.strokes.is_empty() {
            self.export_status = "rien à exporter (aucun stroke)".to_string();
            return;
        }
        match export_inkml_to(std::path::Path::new("exports"), &self.strokes, &self.export_label) {
            Ok(p) => {
                self.export_status = format!("exporté : {}", p.display());
                eprintln!("{}", self.export_status);
            }
            Err(e) => self.export_status = format!("erreur export : {e}"),
        }
    }

    /// Charge un autre modèle dans un thread d'arrière-plan (le téléchargement
    /// HF + le parsing du checkpoint peuvent prendre plusieurs secondes) puis
    /// bascule dessus dès réception.
    fn start_model_load(&mut self, idx: usize, ctx: egui::Context) {
        let src = self.model_sources[idx].clone();
        let cfg = self.cfg.clone();
        let opts = self.decode_opts.clone();
        let (tx, rx) = channel();
        self.model_rx = Some(rx);
        self.model_loading = true;
        self.model_desc = format!("chargement de {} ...", src.label());
        std::thread::spawn(move || {
            let device = candle_core::Device::Cpu;
            let (rec, desc) = build_recognizer(&src, &cfg, opts, &device);
            let _ = tx.send((rec, desc));
            ctx.request_repaint();
        });
    }

    /// Bascule sur le modèle reçu de l'arrière-plan et re-reconnaît ce qui
    /// est à l'écran avec le nouveau modèle.
    fn apply_loaded_model(&mut self, rec: Box<dyn Recognizer>, desc: String) {
        self.model_rx = None;
        self.model_loading = false;
        eprintln!("modèle actif : {desc}");
        self.recognizer = rec;
        self.model_desc = desc;
        if self.strokes.len() > self.finalized {
            self.recognize_window();
        } else if !self.strokes.is_empty() {
            self.recognize_all();
        }
    }

    /// Ajoute un point au stroke en cours (coordonnées relatives au canvas).
    fn push_point(&mut self, p: egui::Pos2, pressure: f32, now: f64, rect: egui::Rect) {
        if let Some(stroke) = &mut self.current {
            stroke.push(Point {
                x: p.x - rect.min.x,
                y: p.y - rect.min.y,
                t: now,
                pressure,
            });
            self.last_draw_time = now;
        }
    }

    /// Termine le stroke en cours et l'archive.
    fn end_stroke(&mut self, now: f64) {
        if let Some(stroke) = self.current.take() {
            if !stroke.is_empty() {
                self.strokes.push(stroke);
                self.last_draw_time = now;
            }
        }
    }

    /// Capture du tracé souris/touch dans le canvas. On lit les événements
    /// bruts — un stylet sous Wayland arrive en `Event::Touch` avec une force
    /// (pression) et peut aussi émettre un `PointerButton` émulé.
    ///
    /// Souris et touch sont traités SÉPARÉMENT, et l'état du bouton souris est
    /// suivi nous-mêmes (`mouse_down`, bouton principal uniquement) plutôt que
    /// via `pointer.any_down()` : après un contact stylet, le release du
    /// PointerButton émulé peut ne jamais arriver, laissant `any_down()` bloqué
    /// à true — la souris traçait alors en continu en hover. Ici, un Touch
    /// Start remet `mouse_down` à false, et la perte du pointeur / du focus
    /// aussi. Un touch ou un contact evdev sans événement depuis plus d'une
    /// seconde est considéré comme terminé (Start sans End, Up manqué).
    fn handle_pointer_input(&mut self, ui: &egui::Ui, rect: egui::Rect, now: f64) {
        let (mut touch_started, mut touch_ended, mut touch_moved, mut pressure, mut touch_pos) =
            (false, false, false, 1.0f32, None);
        let (btn_pressed, btn_released, pointer_gone, pointer_pos, focused) =
            ui.input(|i| {
                let mut bp = false;
                let mut br = false;
                let mut gone = false;
                for ev in &i.events {
                    match ev {
                        egui::Event::PointerButton {
                            button, pressed, ..
                        } if *button == egui::PointerButton::Primary => {
                            if *pressed {
                                bp = true;
                            } else {
                                br = true;
                            }
                        }
                        egui::Event::PointerGone => gone = true,
                        _ => {}
                    }
                }
                (bp, br, gone, i.pointer.latest_pos(), i.focused)
            });
        ui.input(|i| {
            for ev in &i.events {
                if let egui::Event::Touch { phase, pos, force, .. } = ev {
                    match phase {
                        egui::TouchPhase::Start => {
                            touch_started = true;
                            touch_pos = Some(*pos);
                        }
                        egui::TouchPhase::Move => {
                            touch_moved = true;
                            touch_pos = Some(*pos);
                        }
                        egui::TouchPhase::End | egui::TouchPhase::Cancel => {
                            touch_ended = true;
                        }
                    }
                    if let Some(f) = force {
                        pressure = *f;
                    }
                }
            }
        });

        // --- Mise à jour de l'état touch ---
        if touch_started {
            self.touch_active = true;
            // Un contact tactile n'est pas un clic souris : sous Wayland, le
            // stylet émet parfois un PointerButton pressé dont le release
            // n'arrive jamais ; ne jamais prendre ce `down` pour la souris.
            self.mouse_down = false;
            self.last_touch_event = now;
        }
        if touch_moved || touch_ended {
            self.last_touch_event = now;
        }
        if let Some(p) = touch_pos {
            self.last_touch_pos = Some(p);
        }
        if touch_ended {
            self.touch_active = false;
            self.last_touch_pos = None;
        }
        // Sécurité : un touch sans aucun événement depuis > 1 s est un touch
        // resté bloqué (Start sans End) -> on le termine et on archive le stroke.
        if self.touch_active && now - self.last_touch_event > 1.0 {
            self.touch_active = false;
            self.last_touch_pos = None;
            if !self.mouse_down && !self.tablet_active {
                self.end_stroke(now);
            }
        }
        // Idem pour un contact tablette evdev dont le Up aurait été manqué.
        if self.tablet_active && now - self.last_tablet_event > 1.0 {
            self.tablet_active = false;
            if !self.mouse_down && !self.touch_active {
                self.end_stroke(now);
            }
        }

        // --- Mise à jour de l'état bouton principal de la souris ---
        if btn_pressed && !touch_started && !self.touch_active {
            self.mouse_down = true;
        }
        if btn_released {
            self.mouse_down = false;
        }
        // Le pointeur a quitté la fenêtre ou la fenêtre a perdu le focus : un
        // bouton encore "down" chez egui est un résidu (release hors fenêtre) ->
        // on arrête de tracer.
        if pointer_gone || !focused {
            self.mouse_down = false;
        }

        // Démarre un stroke uniquement sur un vrai début d'input (clic souris
        // ou touch start) avec le pointeur dans le canvas.
        let mouse_pressed = btn_pressed && self.mouse_down;
        let start_pos = touch_pos.or(pointer_pos);
        if (mouse_pressed || touch_started) && start_pos.map(|p| rect.contains(p)).unwrap_or(false) {
            self.current = Some(Stroke::new());
        }

        // Souris : tracer UNIQUEMENT si le bouton principal est réellement enfoncé.
        if self.mouse_down && !touch_ended {
            if let Some(p) = pointer_pos {
                if rect.contains(p) {
                    self.push_point(p, 1.0, now, rect);
                }
            }
        }
        // Touch / stylet : tracer uniquement sur les événements touch reçus
        // dans cette frame (pas de point fantôme si le touch est bloqué).
        if self.touch_active && !touch_ended {
            if let Some(p) = touch_pos {
                if rect.contains(p) {
                    self.push_point(p, pressure, now, rect);
                }
            }
        }
        // Termine le stroke sur relâchement / fin de touch, ou si plus aucun
        // input n'est actif (évite un stroke fantôme bloqué). Le contact
        // tablette lu via evdev compte aussi : winit ne lui remonte aucun
        // événement, donc mouse_down/touch_active restent faux pendant le tracé.
        if btn_released
            || touch_ended
            || (!self.mouse_down && !self.touch_active && !self.tablet_active)
        {
            self.end_stroke(now);
        }
    }

    /// Tracé au stylet lu directement via evdev (winit ne remonte pas
    /// toujours le périphérique à egui). Coordonnées normalisées 0..1 de
    /// l'écran : projetées sur le moniteur entier (points écran via
    /// `viewport.monitor_size`), puis rapportées à la position réelle de la
    /// fenêtre sur l'écran (`viewport.inner_rect`). Fenêtré : seule la zone
    /// physique couverte par la fenêtre est dessinable ; plein écran :
    /// `inner_rect.min == (0,0)`, identique à l'ancien comportement.
    fn handle_tablet_events(&mut self, ctx: &egui::Context, rect: egui::Rect, now: f64) {
        let mut tablet_events = Vec::new();
        if let Some(rx) = &self.tablet_rx {
            while let Ok(ev) = rx.try_recv() {
                tablet_events.push(ev);
            }
        }
        if tablet_events.is_empty() {
            return;
        }
        self.record_tablet(&tablet_events);
        self.last_tablet_event = now;
        let (inner_rect, monitor_size) = ctx.input(|i| {
            let vp = i.raw.viewports.get(&i.raw.viewport_id);
            (vp.and_then(|v| v.inner_rect), vp.and_then(|v| v.monitor_size))
        });
        let inner = inner_rect.unwrap_or_else(|| ctx.viewport_rect());
        let monitor = monitor_size.unwrap_or_else(|| inner.size());
        let to_ui =
            |x: f32, y: f32| egui::pos2(x * monitor.x - inner.min.x, y * monitor.y - inner.min.y);
        for ev in tablet_events {
            self.tablet_count += 1;
            match ev {
                TabletEvent::Down { x, y, pressure } => {
                    let p = to_ui(x, y);
                    if rect.contains(p) {
                        self.current = Some(Stroke::new());
                        self.tablet_active = true;
                        self.push_point(p, pressure, now, rect);
                    }
                }
                TabletEvent::Move { x, y, pressure } => {
                    let p = to_ui(x, y);
                    if rect.contains(p) {
                        self.push_point(p, pressure, now, rect);
                    }
                }
                TabletEvent::Up => {
                    self.tablet_active = false;
                    self.end_stroke(now);
                }
            }
        }
    }

    /// Reconnaissance complète (bouton) : tout est figé dans `result`.
    fn recognize_all(&mut self) {
        if self.strokes.is_empty() && self.current.is_none() {
            self.result = "( rien à reconnaître )".to_string();
            return;
        }
        match self.recognizer.recognize_strokes(&self.strokes) {
            Ok(text) => self.result = text,
            Err(e) => self.result = format!("erreur: {e:#}"),
        }
        self.update_debug_img();
        self.finalized = self.strokes.len();
        self.pending_text.clear();
        self.sync_export_label();
    }

    /// Re-reconnaît la fenêtre glissante (strokes non finalisés) et REMPLACE
    /// l'hypothèse en cours : le texte se corrige en direct.
    fn recognize_window(&mut self) {
        let pending = &self.strokes[self.finalized..];
        if pending.is_empty() {
            self.pending_text.clear();
            return;
        }
        match self.recognizer.recognize_strokes(pending) {
            Ok(text) => self.pending_text = text,
            Err(e) => self.pending_text = format!("erreur: {e:#}"),
        }
        self.update_debug_img();
        self.sync_export_label();
    }

    /// Fige la fenêtre courante dans le texte finalisé.
    fn finalize_window(&mut self) {
        if !self.pending_text.is_empty() {
            if !self.result.is_empty() {
                self.result.push(' ');
            }
            self.result.push_str(&self.pending_text);
        }
        self.finalized = self.strokes.len();
        self.pending_text.clear();
        self.sync_export_label();
    }

    /// En mode debug, récupère l'image envoyée au modèle pour l'afficher.
    fn update_debug_img(&mut self) {
        if self.debug {
            self.debug_img = self.recognizer.last_input_image().map(|g| {
                egui::ColorImage::from_gray(
                    [g.width() as usize, g.height() as usize],
                    g.as_raw(),
                )
            });
        }
    }
}

impl eframe::App for InkApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.ui_inner(ui);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.flush_recorder();
    }
}

impl Drop for InkApp {
    fn drop(&mut self) {
        self.flush_recorder();
    }
}

impl InkApp {
    /// Contenu complet de la fenêtre (extrait de `eframe::App::ui` pour être
    /// appelable dans un `egui::Context` de test).
    fn ui_inner(&mut self, ui: &mut egui::Ui) {
        // --record-input : capture la frame courante avant tout traitement.
        self.record_frame(ui.ctx());

        // Modèle chargé en arrière-plan : on bascule dessus.
        let loaded = self.model_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some((rec, desc)) = loaded {
            self.apply_loaded_model(rec, desc);
        }
        if self.model_loading {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(150));
        }

        egui::Panel::top("bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("ecriture-rs");
                if ui.button("Reconnaître").clicked() {
                    self.recognize_all();
                }
                if ui.button("Effacer").clicked() {
                    self.strokes.clear();
                    self.current = None;
                    self.result.clear();
                    self.pending_text.clear();
                    self.finalized = 0;
                }
                if ui.button("Exporter InkML").clicked() {
                    self.do_export();
                }
                if !self.export_status.is_empty() {
                    let ok = self.export_status.starts_with("exporté");
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&self.export_status).small().color(if ok {
                                egui::Color32::DARK_GREEN
                            } else {
                                egui::Color32::RED
                            }),
                        )
                        .truncate(),
                    );
                }
                ui.label(format!("{} strokes", self.strokes.len()));
                ui.checkbox(&mut self.debug, "debug");
                ui.separator();
                // Sélecteur de modèle + indication du modèle actif, à droite.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let selected = if self.model_loading {
                        "chargement…".to_string()
                    } else {
                        self.model_sources[self.model_index].label()
                    };
                    let mut new_index = self.model_index;
                    egui::ComboBox::from_id_salt("model_choice")
                        .width(360.0)
                        .selected_text(selected)
                        .show_ui(ui, |ui| {
                            for (i, s) in self.model_sources.iter().enumerate() {
                                ui.selectable_value(&mut new_index, i, s.label());
                            }
                        });
                    if !self.model_loading && new_index != self.model_index {
                        self.model_index = new_index;
                        self.start_model_load(new_index, ui.ctx().clone());
                    }
                    let desc_color = if self.model_loading {
                        egui::Color32::GOLD
                    } else {
                        egui::Color32::GRAY
                    };
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&self.model_desc).small().color(desc_color),
                        )
                        .truncate(),
                    );
                });
            });
        });

        // Panneau du texte reconnu : redimensionnable (tirer la bordure du
        // haut), taille par défaut generous, texte en grand (28 pt).
        egui::Panel::bottom("result")
            .resizable(true)
            .default_size(200.0)
            .min_size(120.0)
            .max_size(500.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Texte reconnu").strong().size(17.0));
                    ui.label(
                        egui::RichText::new(format!("modèle : {}", self.model_desc))
                            .small()
                            .color(egui::Color32::GRAY),
                    );
                });
                // Vérité terrain pour l'export InkML : pré-remplie avec le
                // texte reconnu, corrigible à la main avant export.
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("vérité terrain :")
                            .small()
                            .color(egui::Color32::GRAY),
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut self.export_label)
                            .desired_width(ui.available_width() - 16.0)
                            .font(egui::TextStyle::Monospace),
                    );
                });
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if self.result.is_empty() && self.pending_text.is_empty() {
                            ui.label(
                                egui::RichText::new(
                                    "(écrivez dans la zone centrale : le texte reconnu s'affiche ici)",
                                )
                                .monospace()
                                .size(26.0)
                                .color(egui::Color32::GRAY),
                            );
                        } else {
                            // Texte finalisé en noir, hypothèse de la fenêtre en cours en gris.
                            ui.label(egui::RichText::new(&self.result).monospace().size(28.0));
                            if !self.pending_text.is_empty() {
                                ui.label(
                                    egui::RichText::new(&self.pending_text)
                                        .monospace()
                                        .size(24.0)
                                        .color(egui::Color32::GRAY),
                                );
                            }
                        }
                    });
                if self.debug {
                    ui.label(egui::RichText::new(&self.debug_info).monospace().small());
                }
            });

        egui::CentralPanel::default().show(ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::drag());

            // Fond blanc du papier.
            painter.rect_filled(response.rect, 0.0, egui::Color32::WHITE);

            // Rendu des strokes terminés.
            for stroke in &self.strokes {
                draw_stroke(&painter, stroke, response.rect);
            }
            if let Some(stroke) = &self.current {
                draw_stroke(&painter, stroke, response.rect);
            }

            // Capture du tracé (souris + touch) dans le canvas.
            let now = ui.input(|i| i.time);
            let rect = response.rect;
            self.handle_pointer_input(ui, rect, now);

            // Tracé au stylet lu directement via evdev (winit ne remonte pas
            // toujours le périphérique à egui).
            self.handle_tablet_events(ui.ctx(), rect, now);

            // Mode debug : remonte ce que egui reçoit comme entrée.
            if self.debug {
                self.debug_info = ui.input(|i| {
                    let evs: Vec<String> = i
                        .events
                        .iter()
                        .filter_map(|ev| match ev {
                            egui::Event::PointerButton { button, pressed, pos, .. } => Some(
                                format!(
                                    "button {:?} {} @ {:.0},{:.0}",
                                    button,
                                    if *pressed { "down" } else { "up" },
                                    pos.x,
                                    pos.y
                                ),
                            ),
                            egui::Event::Touch { phase, pos, force, .. } => Some(format!(
                                "touch {:?} @ {:.0},{:.0} force={:?}",
                                phase, pos.x, pos.y, force
                            )),
                            _ => None,
                        })
                        .collect();
                    format!(
                        "down={} pressed={} released={} any_touches={} pos={:?} mouse_down={} touch_active={} tablet_active={}\n{}",
                        i.pointer.any_down(),
                        i.pointer.any_pressed(),
                        i.pointer.any_released(),
                        i.any_touches(),
                        i.pointer.latest_pos(),
                        self.mouse_down,
                        self.touch_active,
                        self.tablet_active,
                        evs.join("\n")
                    )
                });
                self.debug_info
                    .push_str(&format!("\ntablet events: {}", self.tablet_count));
            }

            // Tant qu'un stroke est ouvert, on force le repaint : avec le
            // stylet lu en evdev, aucun événement winit n'arrive à egui,
            // donc egui ne repeint pas tout seul et la trace n'apparaîtrait
            // qu'à la fin du stroke.
            if self.current.is_some() {
                ui.ctx().request_repaint();
            }
        });

        // Fenêtre glissante : après une pause de tracé, on re-reconnaît les
        // strokes non finalisés (l'hypothèse se corrige en direct) ; après une
        // pause plus longue (ou si la fenêtre dépasse WINDOW_MAX_STROKES), on
        // fige la fenêtre dans le texte finalisé.
        let now = ui.ctx().input(|i| i.time);
        let pending = self.strokes.len().saturating_sub(self.finalized);
        if pending > 0 {
            let pause = now - self.last_draw_time;
            if pause > FINALIZE_PAUSE_SECS || pending > WINDOW_MAX_STROKES {
                // fige les plus vieux strokes pour ne garder que la fenêtre
                if pending > WINDOW_MAX_STROKES {
                    let keep = WINDOW_MAX_STROKES / 2;
                    let split = self.strokes.len() - keep;
                    let window: Vec<Stroke> = self.strokes[..split].to_vec();
                    if let Ok(text) = self.recognizer.recognize_strokes(&window) {
                        if !text.is_empty() {
                            if !self.result.is_empty() {
                                self.result.push(' ');
                            }
                            self.result.push_str(&text);
                        }
                    }
                    self.finalized = split;
                } else {
                    self.finalize_window();
                }
            } else if pause > DEBOUNCE_SECS {
                self.recognize_window();
            }
            // On reprogramme un rafraîchissement pour surveiller la pause.
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }

        // Mode debug : image envoyée au modèle, affichée en haut à droite (x4).
        if self.debug && let Some(ci) = &self.debug_img {
            let tex = ui
                .ctx()
                .load_texture("debug_input", ci.clone(), egui::TextureOptions::NEAREST);
            let size = egui::vec2(ci.width() as f32 * 4.0, ci.height() as f32 * 4.0);
            egui::Window::new(format!("image modèle {}x{}", ci.width(), ci.height()))
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-8.0, 8.0))
                .show(ui.ctx(), |ui| {
                    ui.add(egui::Image::new(&tex).fit_to_exact_size(size));
                });
        }
    }
}

/// Rejoue un fichier d'input enregistré avec --record-input dans un
/// `egui::Context` headless (sans fenêtre) : chaque ligne `Frame` est passée
/// à `ctx.run_ui`, chaque ligne `Tablet` est envoyée dans le channel evdev de
/// l'app puis consommée par une frame. Retourne l'app résultante et le nombre
/// de frames rejouées (pour les assertions des tests).
fn replay_file(path: &str, cfg: &Config) -> anyhow::Result<(InkApp, u64)> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path).with_context(|| format!("ouverture de {path}"))?;
    let src = initial_source(cfg);
    let (tx, rx) = channel();
    let mut app = InkApp::with_tablet_rx(
        Box::new(MockRecognizer),
        format!("replay {path} (reconnaissance simulée)"),
        model_choices(&src),
        0,
        cfg.clone(),
        DecodeOptions::default(),
        Some(rx),
    );
    let ctx = egui::Context::default();
    let mut last_time = 0.0f64;
    let mut last_screen: Option<egui::Rect> = None;
    let mut n_frames = 0u64;
    for (lineno, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("lecture ligne {}", lineno + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: RecordedLine =
            serde_json::from_str(&line).with_context(|| format!("parse ligne {}", lineno + 1))?;
        match rec {
            RecordedLine::Frame(raw) => {
                if let Some(t) = raw.time {
                    last_time = t;
                }
                if raw.screen_rect.is_some() {
                    last_screen = raw.screen_rect;
                }
                let mut out = ctx.run_ui(raw, |ui| app.ui_inner(ui));
                out.textures_delta.clear();
                n_frames += 1;
            }
            RecordedLine::Tablet(evs) => {
                for ev in evs {
                    let _ = tx.send(ev);
                }
                // Frame "vide" (mêmes time/screen_rect) pour consommer le channel.
                let raw = egui::RawInput {
                    screen_rect: last_screen,
                    time: Some(last_time),
                    ..Default::default()
                };
                let mut out = ctx.run_ui(raw, |ui| app.ui_inner(ui));
                out.textures_delta.clear();
                n_frames += 1;
            }
        }
    }
    Ok((app, n_frames))
}

/// --replay-input : rejoue un enregistrement et affiche le résumé des strokes.
fn run_replay(path: &str, cfg: &Config) -> anyhow::Result<()> {
    let (app, n_frames) = replay_file(path, cfg)?;
    println!("replay: {n_frames} frames depuis {path}");
    println!("strokes archivés: {}", app.strokes.len());
    for (i, s) in app.strokes.iter().enumerate() {
        println!("  stroke {i}: {} points", s.points.len());
    }
    if let Some(cur) = &app.current {
        println!("stroke en cours (non archivé): {} points", cur.points.len());
    }
    println!("result: {:?}", app.result);
    println!("pending: {:?}", app.pending_text);
    Ok(())
}

/// Exporte les strokes en InkML dans `dir` (créé si besoin), avec le label
/// (vérité terrain). Retourne le chemin du fichier écrit.
fn export_inkml_to(
    dir: &std::path::Path,
    strokes: &[Stroke],
    label: &str,
) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("ink-{}.inkml", utc_slug()));
    std::fs::write(&path, ink::strokes_to_inkml(strokes, label))?;
    Ok(path)
}

/// Slug UTC lisible : `2026-11-25-161530` (date civile depuis l'epoch Unix).
fn utc_slug() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Date civile depuis le nombre de jours depuis l'epoch (H. Hinnant).
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}-{h:02}{m:02}{s:02}")
}

fn draw_stroke(painter: &egui::Painter, stroke: &Stroke, rect: egui::Rect) {
    if stroke.points.len() < 2 {
        return;
    }
    let points: Vec<egui::Pos2> = stroke
        .points
        .iter()
        .map(|p| egui::pos2(rect.min.x + p.x, rect.min.y + p.y))
        .collect();
    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(2.5, egui::Color32::BLACK),
    ));
}

/// Configuration (args CLI).
#[derive(Clone)]
struct Config {
    /// Checkpoint .pth local ; vocab.json et config.json attendus à côté.
    pth: Option<String>,
    /// URL de base (hors HF) pour télécharger weights + syms.txt PyLaIA dans
    /// le cache local ~/.cache/ecriture-rs/pylaia-rimes/ (GitHub Release, S3...).
    weights_url: Option<String>,
    /// Dossier local contenant weights.ckpt/weights.safetensors + syms.txt.
    weights_dir: Option<String>,
    /// Dépôt Hugging Face téléchargé si --pth absent (modèle par défaut).
    repo: String,
    /// Hauteur d'entrée du modèle (défaut: image_height du config.json, ou
    /// input_shape[1] pour un modèle doctR).
    input_h: Option<u32>,
    /// Largeur d'entrée fixe pour un modèle doctR (défaut: input_shape[2] = 128).
    input_w: Option<u32>,
    norm: Norm,
    /// Périphérique tablette forcé (ex. /dev/input/event9).
    tablet_device: Option<String>,
    /// Lexique de langue pour le rescoring N-best : None = lexique français
    /// embarqué, Some("none") = désactivé, Some(path) = fichier "1 mot/ligne".
    lexicon: Option<String>,
    /// Largeur du beam CTC.
    beam: usize,
    /// Nombre d'hypothèses N-best rescorées par le lexique.
    nbest: usize,
    /// Poids du lexique dans le rescoring (0 = désactivé).
    lm_weight: f64,
    /// Poids du LM de caractères en shallow fusion pendant le beam CTC
    /// (0 = désactivé). Le LM est construit depuis le lexique actif.
    char_lm_weight: f64,
    /// Enregistre tous les événements d'input (frames egui + tablette evdev)
    /// dans un fichier JSONL, rejouable avec --replay-input.
    record_input: Option<String>,
    /// Rejoue un fichier --record-input en mode headless (sans fenêtre) et
    /// affiche le résumé des strokes produits.
    replay_input: Option<String>,
    /// Classifieur de chiffres isolés (LR softmax MNIST) : Auto = poids
    /// embarqués, Off = désactivé, Path = poids custom (.f32 brut ou .json).
    digits: DigitsMode,
    /// Seuil de confiance softmax du classifieur de chiffres pour qu'un
    /// glyphe isolé soit reconnu comme chiffre (sinon repli sur le CRNN).
    digits_threshold: f32,
    /// Backend de rendu : wgpu (Vulkan, défaut) ou glow (OpenGL, recommandé
    /// sur les vieux iGPU Intel Haswell où le support Vulkan de Mesa est
    /// incomplet -> warning + risques de crash).
    renderer: RendererChoice,
}

/// Source du classifieur de chiffres isolés (régression logistique MNIST).
#[derive(Clone, Debug, PartialEq)]
enum DigitsMode {
    /// Poids embarqués dans le binaire (assets/digits-lr.f32).
    Auto,
    /// Classifieur désactivé (--digits off) : que le CRNN.
    Off,
    /// Poids chargés depuis un fichier (.f32 brut 31400 octets ou .json).
    Path(String),
}

/// Backend de rendu egui/eframe choisi en CLI.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RendererChoice {
    Wgpu,
    Glow,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            pth: None,
            weights_url: None,
            weights_dir: None,
            repo: "Sachin-0001/crnn-iam-handwriting".to_string(),
            input_h: None,
            input_w: None,
            norm: Norm::NegOneOne,
            tablet_device: None,
            lexicon: None,
            beam: 16,
            nbest: 8,
            lm_weight: 2.0,
            char_lm_weight: 0.4,
            record_input: None,
            replay_input: None,
            digits: DigitsMode::Auto,
            digits_threshold: 0.85,
            renderer: RendererChoice::Wgpu,
        }
    }
}

/// Source d'un modèle : dépôt Hugging Face, checkpoint local, ou URL de poids
/// custom. Détermine le modèle au démarrage (CLI) et alimente le menu déroulant
/// de l'UI pour basculer de modèle une fois l'application lancée.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ModelSource {
    /// Dépôt Hugging Face, téléchargé dans le cache HF.
    Repo(String),
    /// Checkpoint local ; le backend est détecté depuis les fichiers à côté
    /// (syms.txt -> PyLaIA, arch/vocab dans config.json -> doctR, sinon Sachin).
    Local(PathBuf),
    /// URL de base pour télécharger weights + syms.txt PyLaIA dans le cache local.
    Url(String),
    /// Classifieur de chiffres isolés 0-9 (LR softmax MNIST embarqué,
    /// src/digits.rs) : reconnaît un glyphe unique, pas une ligne de texte.
    Digits,
}

impl ModelSource {
    /// Libellé affiché dans le menu déroulant de l'UI.
    fn label(&self) -> String {
        match self {
            ModelSource::Repo(r) => {
                if r.contains("pylaia-rimes") {
                    format!("{r} · PyLaIA · français (RIMES)")
                } else if r.contains("french") {
                    format!("{r} · doctR · français")
                } else if r.contains("iam") {
                    format!("{r} · CRNN · anglais (IAM)")
                } else {
                    r.clone()
                }
            }
            ModelSource::Local(p) => {
                let pylaia = p
                    .parent()
                    .map(|d| d.join("syms.txt").exists())
                    .unwrap_or(false);
                if pylaia {
                    format!("local {} · PyLaIA · français", p.display())
                } else {
                    format!("local {}", p.display())
                }
            }
            ModelSource::Url(u) => format!("poids distants {u} · PyLaIA · français"),
            ModelSource::Digits => "chiffres isolés 0-9 · LR softmax MNIST (embarqué)".to_string(),
        }
    }
}

/// Source initiale du modèle, déduite de la configuration CLI.
fn initial_source(cfg: &Config) -> ModelSource {
    if let Some(p) = &cfg.pth {
        ModelSource::Local(PathBuf::from(p))
    } else if let Some(u) = &cfg.weights_url {
        ModelSource::Url(u.clone())
    } else if let Some(d) = &cfg.weights_dir {
        let d = PathBuf::from(d);
        let w = ["weights.safetensors", "weights.ckpt", "weights.pth"]
            .iter()
            .map(|w| d.join(w))
            .find(|p| p.exists())
            .unwrap_or_else(|| d.join("weights.ckpt"));
        ModelSource::Local(w)
    } else if let Some((w, _syms)) = model::find_local_pylaia() {
        eprintln!("poids PyLaIA trouvés localement : {}", w.display());
        ModelSource::Local(w)
    } else {
        ModelSource::Repo(cfg.repo.clone())
    }
}

/// Modèles proposés dans le menu de l'UI : les 3 dépôts connus, plus la source
/// initiale (locale) en tête si elle n'en fait pas déjà partie.
fn model_choices(initial: &ModelSource) -> Vec<ModelSource> {
    let mut v = vec![
        ModelSource::Repo("Sachin-0001/crnn-iam-handwriting".to_string()),
        ModelSource::Repo("Teklia/pylaia-rimes".to_string()),
        ModelSource::Repo("Felix92/doctr-torch-crnn-mobilenet-v3-large-french".to_string()),
        ModelSource::Digits,
    ];
    if !v.contains(initial) {
        v.insert(0, initial.clone());
    }
    v
}

/// Résout les fichiers du modèle pour une source puis construit le
/// reconnaisseur CRNN (backend détecté depuis les fichiers à côté).
/// Retourne le reconnaisseur + une description affichable dans l'UI.
fn try_build_recognizer(
    src: &ModelSource,
    cfg: &Config,
    opts: DecodeOptions,
    device: &candle_core::Device,
) -> anyhow::Result<(CrnnRecognizer, String)> {
    // Priorité de résolution d'une source : local > URL (téléchargée dans le
    // cache local) > dépôt HF. Pour PyLaIA, "vocab_json"/"config_json"
    // pointent vers syms.txt (la détection du backend se fait via syms.txt).
    let (pth, vocab_json, config_json): (PathBuf, PathBuf, PathBuf) = match src {
        ModelSource::Local(p) => {
            let dir = p.parent().map(|d| d.to_path_buf()).unwrap_or_default();
            (p.clone(), dir.join("vocab.json"), dir.join("config.json"))
        }
        ModelSource::Url(base) => {
            let dir = model::cache_dir().join("pylaia-rimes");
            let paths = model::download_weights_from_url(
                base,
                &dir,
                &["weights.safetensors", "weights.ckpt", "syms.txt"],
            )
            .with_context(|| format!("téléchargement des poids depuis {base}"))?;
            let syms = dir.join("syms.txt");
            let weights = paths
                .iter()
                .find(|p| p.file_name().map(|n| n == "weights.safetensors").unwrap_or(false))
                .or_else(|| {
                    paths
                        .iter()
                        .find(|p| p.file_name().map(|n| n == "weights.ckpt").unwrap_or(false))
                })
                .cloned();
            match weights {
                Some(w) if syms.exists() => (w, syms.clone(), syms),
                _ => anyhow::bail!("{base} : fichiers téléchargés incomplets (weights + syms.txt)"),
            }
        }
        ModelSource::Repo(repo) => model::download_model_files(repo)
            .with_context(|| format!("téléchargement HF du dépôt {repo}"))?,
        // Le backend chiffres isolés n'a pas de checkpoint : géré directement
        // par build_recognizer (DigitRecognizer).
        ModelSource::Digits => anyhow::bail!("source Digits sans checkpoint (voir build_recognizer)"),
    };
    let config_text = std::fs::read_to_string(&config_json).unwrap_or_default();
    let mconfig: model::ModelConfig = serde_json::from_str(&config_text).unwrap_or_default();
    // Backend PyLaIA si syms.txt est à côté du checkpoint (ex. Teklia/pylaia-rimes).
    let pylaia_syms = pth
        .parent()
        .map(|d| d.join("syms.txt"))
        .filter(|p| p.exists());
    // Hauteur d'entrée : --input-h > 128 (PyLaIA) > image_height (Sachin) > input_shape[1] (doctr).
    let input_h = cfg.input_h.unwrap_or_else(|| {
        if pylaia_syms.is_some() {
            128
        } else {
            mconfig
                .image_height
                .or_else(|| mconfig.input_shape.as_ref().and_then(|s| s.get(1).copied().map(|v| v as u32)))
                .unwrap_or(32)
        }
    });
    // Largeur fixe pour doctR (Sachin garde le ratio, pas de largeur fixe).
    let input_w = cfg.input_w.unwrap_or_else(|| {
        mconfig
            .input_shape
            .as_ref()
            .and_then(|s| s.get(2).copied().map(|v| v as u32))
            .unwrap_or(128)
    });
    let backend = if pylaia_syms.is_some() {
        "pylaia"
    } else if mconfig.is_doctr() {
        "doctr"
    } else {
        "sachin"
    };
    let built: anyhow::Result<CrnnRecognizer> = if let Some(syms) = &pylaia_syms {
        CrnnRecognizer::with_pylaia(&pth, syms, input_h, opts, device)
    } else if mconfig.is_doctr() {
        let dcfg: doctr::DoctrConfig = serde_json::from_str(&config_text)
            .context("config.json doctR invalide")?;
        CrnnRecognizer::with_doctr(&pth, &dcfg, 128, input_h, input_w, opts, device)
    } else {
        CrnnRecognizer::with_options(
            &pth,
            &vocab_json,
            mconfig.num_classes.unwrap_or(80),
            mconfig.hidden_size.unwrap_or(256),
            input_h,
            cfg.norm,
            opts,
            device,
        )
    };
    // Les constructeurs (with_pylaia/with_doctr/with_options) ajoutent déjà
    // un contexte avec le chemin du checkpoint.
    let recognizer = built?;
    let desc = format!(
        "{} · backend={} · h={} w={} · norm={:?}",
        src.label(),
        backend,
        input_h,
        input_w,
        cfg.norm
    );
    Ok((recognizer, desc))
}

/// Charge le classifieur de chiffres isolés selon la config CLI :
/// Auto = poids embarqués, Path = fichier custom (repli embarqué si illisible),
/// Off = None (pas de classifieur, que le CRNN).
fn load_digit_classifier(cfg: &Config) -> Option<DigitClassifier> {
    match &cfg.digits {
        DigitsMode::Off => None,
        DigitsMode::Auto => Some(DigitClassifier::embedded()),
        DigitsMode::Path(p) => match DigitClassifier::load(std::path::Path::new(p)) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("poids chiffres {p} illisibles ({e:#}) : poids embarqués (MNIST)");
                Some(DigitClassifier::embedded())
            }
        },
    }
}

/// Construit le reconnaisseur d'une source ; en cas d'échec, repli sur le
/// MockRecognizer et la description contient l'erreur. La source `Digits`
/// construit un `DigitRecognizer` (chiffres isolés) ; les autres sources CRNN
/// sont enveloppées dans un `HybridRecognizer` qui route les glyphes isolés
/// vers le classifieur LR-MNIST quand il est actif (--digits).
fn build_recognizer(
    src: &ModelSource,
    cfg: &Config,
    opts: DecodeOptions,
    device: &candle_core::Device,
) -> (Box<dyn Recognizer>, String) {
    if matches!(src, ModelSource::Digits) {
        return match load_digit_classifier(cfg) {
            Some(clf) => (
                Box::new(DigitRecognizer::new(clf)),
                src.label(),
            ),
            None => (
                Box::new(MockRecognizer),
                format!("{} · désactivé (--digits off) → reconnaissance simulée", src.label()),
            ),
        };
    }
    let digits = load_digit_classifier(cfg);
    match try_build_recognizer(src, cfg, opts, device) {
        Ok((r, mut desc)) => match digits {
            Some(clf) => {
                desc.push_str(&format!(" · +chiffres LR-MNIST seuil {:.2}", cfg.digits_threshold));
                (
                    Box::new(HybridRecognizer::new(Box::new(r), Some(clf), cfg.digits_threshold)),
                    desc,
                )
            }
            None => (Box::new(r), desc),
        },
        Err(e) => (
            Box::new(MockRecognizer),
            format!("{} · ERREUR ({e:#}) → reconnaissance simulée", src.label()),
        ),
    }
}

fn usage_exit() -> ! {
    eprintln!(
        "usage: ecriture-rs [--pth FICHIER] [--repo owner/name] [--input-h N] [--input-w N]\n\
         \x20         [--norm 01|neg1] [--tablet-device /dev/input/eventN]\n\
         \x20         [--lexicon PATH|none] [--beam N] [--nbest N] [--lm-weight F] [--char-lm-weight F]\n\
         --char-lm-weight F : poids du LM de caractères (uni/bigrammes, construit depuis\n\
         \x20         le lexique) en shallow fusion PENDANT le beam CTC, inspiré de\n\
         \x20         githubharald/CTCDecoder ; 0 = désactivé (défaut 0.4).\n\
         \x20         [--digits auto|off|PATH] [--digits-threshold F]\n\
         \x20         [--record-input events.jsonl] [--replay-input events.jsonl]\n\
         --digits auto|off|PATH : classifieur de chiffres isolés (LR softmax MNIST,\n\
         \x20         poids embarqués assets/digits-lr.f32) ; un glyphe isolé dont la\n\
         \x20         confiance >= --digits-threshold (défaut 0.85) est reconnu comme\n\
         \x20         chiffre 0-9, sinon repli sur le CRNN. off = désactivé,\n\
         \x20         PATH = poids custom (.f32 brut ou .json). Le backend \"chiffres\n\
         \x20         isolés\" du menu déroulant reconnaît un glyphe unique avec ce LR.\n\
         --record-input FICHIER : enregistre tous les événements d'input (souris,\n\
         \x20         touch, tablette evdev) en JSONL pendant la session.\n\
         --replay-input FICHIER : rejoue un enregistrement en headless (sans fenêtre,\n\
         \x20         modèle simulé) et affiche les strokes produits (rejouer un bug en CI).\n\
         --renderer glow|wgpu : backend de rendu (défaut wgpu/Vulkan ; glow=OpenGL,\n\
         \x20         recommandé sur vieux iGPU Intel Haswell où Mesa Vulkan est\n\
         \x20         incomplet -> warning + crashs ; env ECRITURE_RENDERER=glow).\n\
         défauts: --repo Sachin-0001/crnn-iam-handwriting (téléchargé via Hugging Face),\n\
         \x20         --input-h depuis config.json, --norm neg1, --lexicon embarqué (français),\n\
         \x20         --beam 16, --nbest 8, --lm-weight 2.0, --char-lm-weight 0.4,\n\
         \x20         tablette: auto-détection (ou ECRITURE_TABLET_DEVICE)\n\
         \n\
         --pth FICHIER : checkpoint local, format détecté depuis les fichiers à côté :\n\
         \x20         - syms.txt présent -> PyLaIA (ex. modèle FR Teklia/pylaia-rimes) :\n\
         \x20           cargo run --release -- --pth ~/modeles/pylaia-rimes/weights.ckpt\n\
         \x20         - config.json avec arch/vocab -> doctR (pytorch_model.bin + config.json)\n\
         \x20         - sinon Sachin-0001 (model.pth + vocab.json + config.json)\n\
         --repo owner/name : téléchargé depuis Hugging Face (cache auto), format détecté :\n\
         \x20         - syms.txt dans le dépôt -> PyLaIA : --repo Teklia/pylaia-rimes (FR, weights.ckpt auto)\n\
         \x20         - config.json arch/vocab -> doctR : --repo Felix92/doctr-torch-crnn-mobilenet-v3-large-french\n\
         \x20         - sinon Sachin-0001/crnn-iam-handwriting (défaut, anglais)\n\
         \n\
         Partage des poids PyLaIA hors HF (modèle non distribué sur HF en Rust) :\n\
         \x20 --weights-url BASE  : télécharge BASE/weights.ckpt + BASE/syms.txt (ou\n\
         \x20                       weights.safetensors) dans ~/.cache/ecriture-rs/pylaia-rimes/\n\
         \x20                       (GitHub Release, S3, serveur perso...) au 1er lancement\n\
         \x20 --weights-dir DIR   : dossier local contenant weights.* + syms.txt\n\
         \x20 sans option           : cherche weights.* + syms.txt à côté du binaire,\n\
         \x20                       dans ./ et dans le cache local (zip distribué avec les poids)\n\
         Priorité : --pth > --weights-url > --weights-dir > auto-local > --repo"
    );
    std::process::exit(2)
}

fn parse_args() -> Config {
    parse_args_from(std::env::args().skip(1))
}

/// Parse une séquence d'arguments (sans le nom du programme) ; extrait de
/// `parse_args` pour être testable.
fn parse_args_from(args: impl Iterator<Item = String>) -> Config {
    let mut cfg = Config::default();
    let mut renderer_set = false;
    let mut args = args;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pth" => cfg.pth = Some(args.next().unwrap_or_else(|| usage_exit())),
            "--weights-url" => cfg.weights_url = Some(args.next().unwrap_or_else(|| usage_exit())),
            "--weights-dir" => cfg.weights_dir = Some(args.next().unwrap_or_else(|| usage_exit())),
            "--repo" => cfg.repo = args.next().unwrap_or_else(|| usage_exit()),
            "--input-h" => {
                cfg.input_h = Some(
                    args.next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| usage_exit()),
                )
            }
            "--input-w" => {
                cfg.input_w = Some(
                    args.next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| usage_exit()),
                )
            }
            "--norm" => {
                cfg.norm = match args.next().as_deref() {
                    Some("neg1") => Norm::NegOneOne,
                    Some("01") => Norm::ZeroOne,
                    _ => usage_exit(),
                }
            }
            "--tablet-device" => {
                cfg.tablet_device = Some(args.next().unwrap_or_else(|| usage_exit()))
            }
            "--lexicon" => cfg.lexicon = Some(args.next().unwrap_or_else(|| usage_exit())),
            "--beam" => {
                cfg.beam = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage_exit())
            }
            "--nbest" => {
                cfg.nbest = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage_exit())
            }
            "--lm-weight" => {
                cfg.lm_weight = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage_exit())
            }
            "--char-lm-weight" => {
                cfg.char_lm_weight = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage_exit())
            }
            "--record-input" => {
                cfg.record_input = Some(args.next().unwrap_or_else(|| usage_exit()))
            }
            "--replay-input" => {
                cfg.replay_input = Some(args.next().unwrap_or_else(|| usage_exit()))
            }
            "--digits" => {
                cfg.digits = match args.next().as_deref() {
                    Some("auto") => DigitsMode::Auto,
                    Some("off") | Some("none") => DigitsMode::Off,
                    Some(p) => DigitsMode::Path(p.to_string()),
                    None => usage_exit(),
                }
            }
            "--digits-threshold" => {
                cfg.digits_threshold = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage_exit())
            }
            "--renderer" => {
                cfg.renderer = match args.next().as_deref() {
                    Some("glow") => RendererChoice::Glow,
                    Some("wgpu") => RendererChoice::Wgpu,
                    _ => usage_exit(),
                };
                renderer_set = true;
            }
            "-h" | "--help" => usage_exit(),
            other => {
                eprintln!("argument inconnu: {other}");
                usage_exit();
            }
        }
    }
    // --renderer a priorité ; sinon variable d'env ECRITURE_RENDERER=glow|wgpu.
    if !renderer_set {
        if let Ok(r) = std::env::var("ECRITURE_RENDERER") {
            match r.as_str() {
                "glow" => cfg.renderer = RendererChoice::Glow,
                "wgpu" => cfg.renderer = RendererChoice::Wgpu,
                _ => eprintln!("ECRITURE_RENDERER={r} invalide (glow|wgpu), ignoré"),
            }
        }
    }
    cfg
}

fn main() -> eframe::Result<()> {
    let cfg = parse_args();

    // --replay-input : rejoue un enregistrement d'input en headless et quitte
    // (pas de fenêtre, pas de modèle téléchargé : reconnaissance simulée).
    if let Some(path) = &cfg.replay_input {
        return run_replay(path, &cfg)
            .map_err(|e| eframe::Error::AppCreation(format!("{e:#}").into()));
    }

    // Lexique de langue pour le rescoring N-best : embarqué (français) par
    // défaut, fichier local avec --lexicon PATH, désactivé avec --lexicon none.
    let lexicon = match &cfg.lexicon {
        None => Some(lexicon::Lexicon::embedded_fr()),
        Some(p) if p == "none" || p == "-" => None,
        Some(p) => match lexicon::Lexicon::load(std::path::Path::new(p)) {
            Ok(l) => Some(l),
            Err(e) => {
                eprintln!("lexique {p} illisible ({e}) : lexique embarqué (français)");
                Some(lexicon::Lexicon::embedded_fr())
            }
        },
    };
    let lex_info = match &lexicon {
        Some(l) => format!("{} mots", l.len()),
        None => "désactivé".to_string(),
    };
    eprintln!(
        "config: repo={} pth={:?} input_h={:?} input_w={:?} norm={:?} tablet_device={:?} beam={} nbest={} lm_weight={} char_lm_weight={} lexique={} digits={:?} digits_threshold={}",
        cfg.repo,
        cfg.pth,
        cfg.input_h,
        cfg.input_w,
        cfg.norm,
        cfg.tablet_device,
        cfg.beam,
        cfg.nbest,
        cfg.lm_weight,
        cfg.char_lm_weight,
        lex_info,
        cfg.digits,
        cfg.digits_threshold,
    );
    let decode_opts = DecodeOptions {
        beam_size: cfg.beam,
        nbest: cfg.nbest,
        lm_weight: cfg.lm_weight,
        lexicon,
        char_lm_weight: cfg.char_lm_weight,
        char_lm: None, // construit depuis le lexique au chargement du modèle
    };

    // Source du modèle (priorité : --pth > --weights-url > --weights-dir >
    // auto-local > --repo HF) puis construction, avec repli sur le mock.
    let src = initial_source(&cfg);
    eprintln!("source modèle : {}", src.label());
    let device = candle_core::Device::Cpu;
    let (recognizer, model_desc) = build_recognizer(&src, &cfg, decode_opts.clone(), &device);
    eprintln!("modèle actif : {model_desc}");
    let model_sources = model_choices(&src);
    let model_index = model_sources.iter().position(|s| s == &src).unwrap_or(0);

    // Backend de rendu : wgpu (Vulkan) par défaut ; glow (OpenGL) sur les vieux
    // iGPU Intel (Haswell) où Mesa Vulkan est incomplet (--renderer glow ou
    // ECRITURE_RENDERER=glow).
    let renderer = match cfg.renderer {
        RendererChoice::Wgpu => eframe::Renderer::Wgpu,
        RendererChoice::Glow => eframe::Renderer::Glow,
    };
    eprintln!("renderer: {}", renderer);
    let options = eframe::NativeOptions {
        renderer,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 800.0])
            .with_min_inner_size([720.0, 520.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ecriture-rs",
        options,
        Box::new(|cc| {
            Ok(Box::new(InkApp::new(
                recognizer,
                model_desc,
                model_sources,
                model_index,
                cfg.clone(),
                decode_opts,
                cfg.tablet_device.clone(),
                cc.egui_ctx.clone(),
            )))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choix_modeles_inclut_repos_connus_et_source_initiale() {
        let local = ModelSource::Local(PathBuf::from("/tmp/mon-modele/model.pth"));
        let v = model_choices(&local);
        assert_eq!(v[0], local);
        assert!(v.contains(&ModelSource::Repo("Sachin-0001/crnn-iam-handwriting".into())));
        assert!(v.contains(&ModelSource::Repo("Teklia/pylaia-rimes".into())));
        assert!(v.contains(&ModelSource::Repo(
            "Felix92/doctr-torch-crnn-mobilenet-v3-large-french".into()
        )));
        // Le backend chiffres isolés (LR MNIST) est proposé dans le menu.
        assert!(v.contains(&ModelSource::Digits));
        assert!(ModelSource::Digits.label().contains("LR softmax MNIST"));
        // Pas de doublon si la source initiale est déjà un dépôt connu.
        let rimes = ModelSource::Repo("Teklia/pylaia-rimes".into());
        let v2 = model_choices(&rimes);
        assert_eq!(v2.iter().filter(|s| **s == rimes).count(), 1);
        assert_eq!(v2.len(), 4); // 3 repos + Digits
    }

    #[test]
    fn parse_args_digit_options() {
        // défaut : auto + seuil 0.85
        let cfg = parse_args_from([].into_iter());
        assert_eq!(cfg.digits, DigitsMode::Auto);
        assert!((cfg.digits_threshold - 0.85).abs() < 1e-6);
        // --digits off / --digits PATH / --digits-threshold
        let cfg = parse_args_from(["--digits".into(), "off".into()].into_iter());
        assert_eq!(cfg.digits, DigitsMode::Off);
        let cfg = parse_args_from(["--digits".into(), "/tmp/mes-poids.f32".into()].into_iter());
        assert_eq!(cfg.digits, DigitsMode::Path("/tmp/mes-poids.f32".into()));
        let cfg = parse_args_from(
            ["--digits-threshold".into(), "0.5".into(), "--digits".into(), "auto".into()]
                .into_iter(),
        );
        assert!((cfg.digits_threshold - 0.5).abs() < 1e-6);
        assert_eq!(cfg.digits, DigitsMode::Auto);
    }

    #[test]
    fn parse_args_char_lm_weight() {
        let cfg = parse_args_from([].into_iter());
        assert!((cfg.char_lm_weight - 0.4).abs() < 1e-9);
        let cfg = parse_args_from(["--char-lm-weight".into(), "0".into()].into_iter());
        assert_eq!(cfg.char_lm_weight, 0.0);
        let cfg = parse_args_from(["--char-lm-weight".into(), "1.5".into()].into_iter());
        assert!((cfg.char_lm_weight - 1.5).abs() < 1e-9);
    }

    #[test]
    fn load_digit_classifier_selon_mode() {
        let mut cfg = Config::default();
        assert!(load_digit_classifier(&cfg).is_some()); // Auto -> embarqué
        cfg.digits = DigitsMode::Off;
        assert!(load_digit_classifier(&cfg).is_none());
        // Path illisible -> repli sur l'embarqué avec un warning
        cfg.digits = DigitsMode::Path("/nonexistent/digits.f32".into());
        assert!(load_digit_classifier(&cfg).is_some());
    }

    #[test]
    fn build_recognizer_digits_construit_digit_recognizer() {
        let cfg = Config::default();
        let (mut rec, desc) = build_recognizer(
            &ModelSource::Digits,
            &cfg,
            DecodeOptions::default(),
            &candle_core::Device::Cpu,
        );
        assert!(desc.contains("LR softmax MNIST"), "desc={desc}");
        // Un glyphe isolé -> un chiffre 0-9 (pas le mock).
        use ecriture_rs::ink::Point;
        let mut s = ecriture_rs::ink::Stroke::new();
        for i in 0..=20 {
            let r = i as f32 / 20.0;
            s.push(Point {
                x: 50.0,
                y: 10.0 + r * 80.0,
                t: r as f64,
                pressure: 0.8,
            });
        }
        let out = rec.recognize_strokes(&[s]).unwrap();
        assert_eq!(out.chars().count(), 1);
        assert!(out.chars().next().unwrap().is_ascii_digit(), "reçu {out:?}");
    }

    #[test]
    fn source_initiale_suit_la_config() {
        let mut cfg = Config::default();
        if model::find_local_pylaia().is_none() {
            assert_eq!(
                initial_source(&cfg),
                ModelSource::Repo("Sachin-0001/crnn-iam-handwriting".into())
            );
        }
        cfg.pth = Some("/tmp/x/model.pth".into());
        assert_eq!(
            initial_source(&cfg),
            ModelSource::Local(PathBuf::from("/tmp/x/model.pth"))
        );
        cfg.pth = None;
        cfg.weights_dir = Some("/tmp/mes-poids".into());
        // weights.ckpt par défaut si aucun weights.* présent dans le dossier
        assert_eq!(
            initial_source(&cfg),
            ModelSource::Local(PathBuf::from("/tmp/mes-poids/weights.ckpt"))
        );
    }

    #[test]
    fn build_recognizer_repli_sur_mock_si_fichier_absent() {
        let cfg = Config::default();
        let src = ModelSource::Local(PathBuf::from("/nonexistent/model.pth"));
        let (mut rec, desc) =
            build_recognizer(&src, &cfg, DecodeOptions::default(), &candle_core::Device::Cpu);
        assert!(desc.contains("ERREUR"), "desc={desc}");
        assert_eq!(rec.recognize_strokes(&[]).unwrap(), "[reconnaissance simulée]");
    }

    /// Test réel (à lancer explicitement) : construit le reconnaisseur depuis
    /// le modèle PyLaIA local si présent.
    /// `cargo test --release --bin ecriture-rs -- --ignored --nocapture modele_local_pylaia`
    #[test]
    #[ignore]
    fn modele_local_pylaia_se_charge() {
        let home = std::env::var("HOME").unwrap();
        let ckpt = PathBuf::from(&home).join("modeles/pylaia-rimes/weights.ckpt");
        if !ckpt.exists() {
            eprintln!("skip: {} absent", ckpt.display());
            return;
        }
        let cfg = Config::default();
        let src = ModelSource::Local(ckpt);
        let (mut r, desc) = try_build_recognizer(
            &src,
            &cfg,
            DecodeOptions::default(),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        assert!(desc.contains("backend=pylaia"), "desc={desc}");
        assert!(desc.contains("h=128"), "desc={desc}");
        let s = Stroke::new();
        assert!(r.recognize_strokes(&[s]).is_ok());
    }

    // ---- Simulation d'input egui (capture souris/touch du canvas) ----

    fn app_test() -> InkApp {
        let src = ModelSource::Repo("Sachin-0001/crnn-iam-handwriting".into());
        InkApp::with_tablet_rx(
            Box::new(MockRecognizer),
            "mock".to_string(),
            model_choices(&src),
            0,
            Config::default(),
            DecodeOptions::default(),
            None,
        )
    }

    /// Simule une frame egui avec les événements d'input donnés, comme le
    /// fait le CentralPanel de l'app.
    fn run_frame(app: &mut InkApp, ctx: &egui::Context, events: Vec<egui::Event>) {
        run_frame_t(app, ctx, 1.0, events);
    }

    /// Idem avec un temps explicite (les timeouts touch/tablette dépendent du temps).
    fn run_frame_t(app: &mut InkApp, ctx: &egui::Context, time: f64, events: Vec<egui::Event>) {
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            time: Some(time),
            events,
            ..Default::default()
        };
        let mut output = ctx.run_ui(raw, |ui| {
            let (response, _painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::drag());
            let now = ui.input(|i| i.time);
            let rect = response.rect;
            app.handle_pointer_input(ui, rect, now);
            app.handle_tablet_events(ui.ctx(), rect, now);
        });
        // Hors écran, personne n'applique les deltas de textures (fonts...) :
        // on les vide pour éviter le panic au drop.
        output.textures_delta.clear();
    }

    fn ptr_button(pos: egui::Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        }
    }

    fn touch_event(phase: egui::TouchPhase, pos: egui::Pos2) -> egui::Event {
        egui::Event::Touch {
            device_id: egui::TouchDeviceId(0),
            id: egui::TouchId(1),
            phase,
            pos,
            force: Some(1.0),
        }
    }

    #[test]
    fn hover_souris_seul_ne_trace_pas() {
        let ctx = egui::Context::default();
        let mut app = app_test();
        run_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(egui::pos2(400.0, 300.0))]);
        run_frame(&mut app, &ctx, vec![egui::Event::PointerMoved(egui::pos2(500.0, 400.0))]);
        assert!(app.current.is_none(), "un stroke a été créé en hover");
        assert!(app.strokes.is_empty());
    }

    #[test]
    fn clic_maintenu_puis_release_archive_un_stroke() {
        let ctx = egui::Context::default();
        let mut app = app_test();
        run_frame(
            &mut app,
            &ctx,
            vec![
                egui::Event::PointerMoved(egui::pos2(400.0, 300.0)),
                ptr_button(egui::pos2(400.0, 300.0), true),
            ],
        );
        run_frame(
            &mut app,
            &ctx,
            vec![egui::Event::PointerMoved(egui::pos2(450.0, 320.0))],
        );
        assert!(app.current.is_some(), "pas de stroke en cours pendant le clic");
        run_frame(
            &mut app,
            &ctx,
            vec![
                egui::Event::PointerMoved(egui::pos2(460.0, 330.0)),
                ptr_button(egui::pos2(460.0, 330.0), false),
            ],
        );
        assert!(app.current.is_none(), "stroke non terminé au release");
        assert_eq!(app.strokes.len(), 1);
        assert!(app.strokes[0].points.len() >= 2);
    }

    #[test]
    fn touch_actif_ne_fait_pas_tracer_la_souris_en_hover() {
        // Un Touch Start est reçu (contact tactile) ; puis la SOURIS bouge en
        // hover (aucun bouton enfoncé) : elle ne doit rien ajouter au stroke.
        let ctx = egui::Context::default();
        let mut app = app_test();
        run_frame(
            &mut app,
            &ctx,
            vec![touch_event(egui::TouchPhase::Start, egui::pos2(400.0, 300.0))],
        );
        assert!(app.current.is_some(), "le touch start doit créer un stroke");
        let p0 = app.current.as_ref().unwrap().points[0];
        run_frame(
            &mut app,
            &ctx,
            vec![egui::Event::PointerMoved(egui::pos2(700.0, 550.0))],
        );
        let cur = app.current.as_ref().expect("stroke touch en cours");
        let last = *cur.points.last().unwrap();
        assert!(
            (last.x - p0.x).abs() < 0.5 && (last.y - p0.y).abs() < 0.5,
            "la souris a tracé en hover pendant un touch actif: p0={p0:?} last={last:?}"
        );
    }

    #[test]
    fn utc_slug_format_date_heure() {
        let s = utc_slug();
        // "2026-11-25-161530" : 17 caractères, tirets en 4, 7, 10
        assert_eq!(s.len(), 17, "slug={s}");
        assert!(
            s.chars().enumerate().all(|(i, c)| i == 4 || i == 7 || i == 10 || c.is_ascii_digit()),
            "slug={s}"
        );
    }

    #[test]
    fn export_inkml_ecrit_un_fichier_valide() {
        let dir = std::path::Path::new("/tmp/ecriture-rs-test-export");
        let _ = std::fs::remove_dir_all(dir);
        let mut s = Stroke::new();
        s.push(Point { x: 1.0, y: 2.0, t: 0.5, pressure: 0.7 });
        s.push(Point { x: 3.0, y: 4.0, t: 0.6, pressure: 0.9 });
        let path = export_inkml_to(dir, &[s], "héllo <monde>").unwrap();
        assert!(path.exists());
        assert!(path.file_name().unwrap().to_string_lossy().starts_with("ink-"));
        assert!(path.to_string_lossy().ends_with(".inkml"));
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("<trace id=\"0\">1 2 0.5 0.7, 3 4 0.6 0.9</trace>"));
        assert!(content.contains("label=\"héllo &lt;monde&gt;\""));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Écrit un InkML de démo dans /tmp/ecriture-rs-demo (validation externe
    /// du XML, ex. avec un parser InkML).
    /// `cargo test --bin ecriture-rs -- --ignored export_inkml_demo`
    #[test]
    #[ignore]
    fn export_inkml_demo_dans_tmp() {
        let mut s = Stroke::new();
        for i in 0..20 {
            s.push(Point {
                x: i as f32 * 4.0,
                y: 40.0 + (i as f32 * 0.3).sin() * 20.0,
                t: i as f64 * 0.016,
                pressure: 0.8,
            });
        }
        let path =
            export_inkml_to(std::path::Path::new("/tmp/ecriture-rs-demo"), &[s], "démo & <test>")
                .unwrap();
        eprintln!("inkml demo -> {}", path.display());
    }

    /// Bug réel (Wayland) : après un contact stylet, winit envoie Touch Start
    /// (+ parfois un PointerButton pressé émulé) mais le End / le release
    /// peuvent ne jamais arriver. `pointer.any_down()` reste alors bloqué à
    /// true et la souris traçait en continu en hover. Le tracé souris doit
    /// reposer sur un vrai clic (press) de la souris, et le touch bloqué doit
    /// être terminé automatiquement.
    #[test]
    fn apres_stylet_la_souris_ne_trace_plus_en_hover() {
        let ctx = egui::Context::default();
        let mut app = app_test();
        // Frame 1 (t=1.0) : le stylet touche l'écran -> Touch Start + PointerButton
        // pressé émulé dans la même frame.
        run_frame_t(
            &mut app,
            &ctx,
            1.0,
            vec![
                touch_event(egui::TouchPhase::Start, egui::pos2(400.0, 300.0)),
                ptr_button(egui::pos2(400.0, 300.0), true),
            ],
        );
        assert!(app.current.is_some(), "le touch start doit créer un stroke");
        assert!(
            !app.mouse_down,
            "le contact stylet ne doit pas activer le tracé souris"
        );
        // Frame 2 (t=1.1) : le release du stylet n'arrive jamais ; la souris
        // bouge en hover -> aucun point ne doit être ajouté au stroke.
        run_frame_t(
            &mut app,
            &ctx,
            1.1,
            vec![egui::Event::PointerMoved(egui::pos2(600.0, 500.0))],
        );
        let n = app.current.as_ref().map(|s| s.points.len()).unwrap_or(0);
        assert_eq!(
            n, 1,
            "la souris a tracé en hover après le stylet (points={n})"
        );
        // Frame 3 (t=3.0) : le touch bloqué est périmé (> 1 s sans événement),
        // puis un vrai clic souris démarre un nouveau stroke.
        run_frame_t(
            &mut app,
            &ctx,
            3.0,
            vec![
                egui::Event::PointerMoved(egui::pos2(500.0, 400.0)),
                ptr_button(egui::pos2(500.0, 400.0), true),
            ],
        );
        assert!(app.mouse_down, "le vrai clic souris doit activer mouse_down");
        assert!(!app.touch_active, "le touch bloqué doit être terminé");
        assert_eq!(app.strokes.len(), 1, "le stroke du stylet doit être archivé");
        // Frame 4 : maintient du clic, la souris trace.
        run_frame_t(
            &mut app,
            &ctx,
            3.1,
            vec![egui::Event::PointerMoved(egui::pos2(550.0, 420.0))],
        );
        let pts = app.current.as_ref().map(|s| s.points.len());
        assert!(
            pts.unwrap_or(0) >= 2,
            "pas de tracé pendant le clic-maintient (points={pts:?})"
        );
        // Frame 5 : relâchement -> le stroke souris est archivé.
        run_frame_t(
            &mut app,
            &ctx,
            3.2,
            vec![ptr_button(egui::pos2(550.0, 420.0), false)],
        );
        assert!(app.current.is_none(), "le stroke doit être fermé au release");
        assert_eq!(app.strokes.len(), 2);
        // Frame 6 : hover après le relâchement -> rien ne se passe.
        run_frame_t(
            &mut app,
            &ctx,
            3.3,
            vec![egui::Event::PointerMoved(egui::pos2(700.0, 550.0))],
        );
        assert!(app.current.is_none() && app.strokes.len() == 2);
    }

    /// Le pointeur quitte la fenêtre avec le bouton enfoncé (release hors
    /// fenêtre, non reçu) : au retour en hover, la souris ne doit pas tracer.
    #[test]
    fn release_hors_fenetre_ne_laisse_pas_de_trace_continue() {
        let ctx = egui::Context::default();
        let mut app = app_test();
        run_frame(
            &mut app,
            &ctx,
            vec![
                egui::Event::PointerMoved(egui::pos2(400.0, 300.0)),
                ptr_button(egui::pos2(400.0, 300.0), true),
            ],
        );
        assert!(app.mouse_down && app.current.is_some());
        // Le curseur quitte la fenêtre (bouton encore enfoncé) : egui ne voit
        // pas le release, any_down() reste true, mais on coupe mouse_down.
        run_frame(&mut app, &ctx, vec![egui::Event::PointerGone]);
        assert!(!app.mouse_down, "PointerGone doit couper mouse_down");
        // Retour en hover dans le canvas : aucun point ajouté.
        run_frame(
            &mut app,
            &ctx,
            vec![egui::Event::PointerMoved(egui::pos2(500.0, 400.0))],
        );
        assert!(app.current.is_none(), "le stroke aurait dû être fermé");
        assert!(app.strokes.len() <= 1);
    }

    #[test]
    fn touch_start_puis_end_ne_laisse_pas_de_stroke_bloque() {
        let ctx = egui::Context::default();
        let mut app = app_test();
        run_frame(
            &mut app,
            &ctx,
            vec![touch_event(egui::TouchPhase::Start, egui::pos2(400.0, 300.0))],
        );
        run_frame(
            &mut app,
            &ctx,
            vec![touch_event(egui::TouchPhase::End, egui::pos2(400.0, 300.0))],
        );
        assert!(app.current.is_none());
        assert_eq!(app.strokes.len(), 1);
        // Après le End, un hover souris ne doit rien créer de plus.
        run_frame(
            &mut app,
            &ctx,
            vec![egui::Event::PointerMoved(egui::pos2(600.0, 500.0))],
        );
        assert!(app.current.is_none() && app.strokes.len() == 1);
    }

    // ---- Simulation du stylet lu via evdev (channel tablette) ----

    /// Reproduit le flux réel du stylet evdev : le thread de lecture envoie
    /// Down/Move/Up dans le channel entre les frames egui, et AUCUN événement
    /// winit n'arrive à egui pendant le tracé (d'où down=false et
    /// touch_active=false à chaque frame). Le stroke doit rester ouvert entre
    /// les frames et n'être archivé qu'au Up.
    #[test]
    fn stylet_evdev_trace_un_stroke_complet_entre_les_frames() {
        let ctx = egui::Context::default();
        let (tx, rx) = channel();
        let src = ModelSource::Repo("Sachin-0001/crnn-iam-handwriting".into());
        let mut app = InkApp::with_tablet_rx(
            Box::new(MockRecognizer),
            "mock".to_string(),
            model_choices(&src),
            0,
            Config::default(),
            DecodeOptions::default(),
            Some(rx),
        );
        // Frame 1 : Down au centre du canvas 800x600 -> (400, 300).
        tx.send(TabletEvent::Down {
            x: 0.5,
            y: 0.5,
            pressure: 1.0,
        })
        .unwrap();
        run_frame(&mut app, &ctx, vec![]);
        assert!(app.current.is_some(), "le Down doit ouvrir un stroke");
        assert!(app.tablet_active);
        // Frame 2 : Move, sans aucun événement egui (stylet evdev seul).
        tx.send(TabletEvent::Move {
            x: 0.55,
            y: 0.5,
            pressure: 1.0,
        })
        .unwrap();
        run_frame(&mut app, &ctx, vec![]);
        let n = app.current.as_ref().map(|s| s.points.len());
        assert!(
            n.unwrap_or(0) >= 2,
            "le stroke a été fermé prématurément après le Down (points={n:?})"
        );
        // Frame 3 : Move puis Up -> le stroke est archivé avec tous ses points.
        tx.send(TabletEvent::Move {
            x: 0.6,
            y: 0.55,
            pressure: 1.0,
        })
        .unwrap();
        tx.send(TabletEvent::Up).unwrap();
        run_frame(&mut app, &ctx, vec![]);
        assert!(app.current.is_none(), "le Up doit fermer le stroke");
        assert!(!app.tablet_active);
        assert_eq!(app.strokes.len(), 1);
        assert!(
            app.strokes[0].points.len() >= 3,
            "points={}",
            app.strokes[0].points.len()
        );
    }

    /// Test d'intégration : fait tourner le VRAI ui complet de l'app
    /// (`ui_inner`) dans un egui::Context, avec un channel tablette alimenté
    /// entre les frames (Down/Move/Up) et aucun événement winit — comme avec
    /// un vrai stylet lu en evdev. Vérifie que la trace est réellement rendue
    /// (lignes noires dans les shapes de sortie) pendant ET après le stroke.
    #[test]
    fn ui_complete_rend_la_trace_du_stylet_evdev() {
        let ctx = egui::Context::default();
        let (tx, rx) = channel();
        let src = ModelSource::Repo("Sachin-0001/crnn-iam-handwriting".into());
        let mut app = InkApp::with_tablet_rx(
            Box::new(MockRecognizer),
            "mock".to_string(),
            model_choices(&src),
            0,
            Config::default(),
            DecodeOptions::default(),
            Some(rx),
        );
        let raw = |t: f64| egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            time: Some(t),
            ..Default::default()
        };
        let lignes_noires = |out: &egui::FullOutput| {
            out.shapes
                .iter()
                .filter(|cs| {
                    matches!(&cs.shape, egui::Shape::Path(p) if p.stroke.color == egui::epaint::ColorMode::Solid(egui::Color32::BLACK))
                })
                .count()
        };

        // Frame 1 : Down au centre (0.5, 0.5) -> (400, 300) dans le canvas.
        tx.send(TabletEvent::Down {
            x: 0.5,
            y: 0.5,
            pressure: 1.0,
        })
        .unwrap();
        let mut out = ctx.run_ui(raw(1.0), |ui| app.ui_inner(ui));
        out.textures_delta.clear();

        // Frame 2 : une série de Moves, sans aucun événement egui (stylet evdev).
        for i in 1..=10 {
            tx.send(TabletEvent::Move {
                x: 0.5 + 0.02 * i as f32,
                y: 0.5,
                pressure: 1.0,
            })
            .unwrap();
        }
        let mut out = ctx.run_ui(raw(1.1), |ui| app.ui_inner(ui));
        out.textures_delta.clear();
        let pts = app.current.as_ref().map(|s| s.points.len());
        assert!(
            pts.unwrap_or(0) >= 10,
            "le stroke en cours a perdu des points (points={pts:?})"
        );

        // Frame 3 : aucun nouvel événement -> le rendu affiche enfin les points
        // reçus à la frame 2 (le dessin précède le traitement de l'input).
        let mut out = ctx.run_ui(raw(1.2), |ui| app.ui_inner(ui));
        let n = lignes_noires(&out);
        out.textures_delta.clear();
        assert!(
            n >= 1,
            "aucune ligne noire rendue pendant le tracé stylet (n={n})"
        );

        // Frame 4 : Up -> le stroke est archivé mais la trace reste affichée.
        tx.send(TabletEvent::Up).unwrap();
        let mut out = ctx.run_ui(raw(1.3), |ui| app.ui_inner(ui));
        out.textures_delta.clear();
        let mut out = ctx.run_ui(raw(1.4), |ui| app.ui_inner(ui));
        let n = lignes_noires(&out);
        out.textures_delta.clear();
        assert!(n >= 1, "la trace doit rester affichée après le Up (n={n})");
        assert_eq!(app.strokes.len(), 1);
        assert!(
            app.strokes[0].points.len() >= 10,
            "points={}",
            app.strokes[0].points.len()
        );
    }

    /// Fenêtré : le stylet ne doit capturer que la zone physique de l'écran
    /// couverte par la fenêtre, pas l'écran entier. On simule un moniteur
    /// 2560x1440 et une fenêtre 800x600 posée en (1000, 200) sur l'écran.
    #[test]
    fn tablette_fenetre_ne_capture_que_la_zone_physique_de_la_fenetre() {
        let ctx = egui::Context::default();
        let (tx, rx) = channel();
        let src = ModelSource::Repo("Sachin-0001/crnn-iam-handwriting".into());
        let mut app = InkApp::with_tablet_rx(
            Box::new(MockRecognizer),
            "mock".to_string(),
            model_choices(&src),
            0,
            Config::default(),
            DecodeOptions::default(),
            Some(rx),
        );
        // Fenêtre 800x600 (points UI) posée à (1000, 200) sur un moniteur 2560x1440.
        let raw = |t: f64| egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            viewports: [(
                egui::ViewportId::ROOT,
                egui::ViewportInfo {
                    inner_rect: Some(egui::Rect::from_min_size(
                        egui::pos2(1000.0, 200.0),
                        egui::vec2(800.0, 600.0),
                    )),
                    monitor_size: Some(egui::vec2(2560.0, 1440.0)),
                    ..Default::default()
                },
            )]
            .into_iter()
            .collect(),
            time: Some(t),
            ..Default::default()
        };

        // Hors de la fenêtre physique : x=0.1 -> 256 px écran < 1000 -> ignoré.
        tx.send(TabletEvent::Down {
            x: 0.1,
            y: 0.5,
            pressure: 1.0,
        })
        .unwrap();
        let mut out = ctx.run_ui(raw(1.0), |ui| app.ui_inner(ui));
        out.textures_delta.clear();
        assert!(
            app.current.is_none(),
            "hors fenêtre : aucun stroke ne doit s'ouvrir"
        );
        tx.send(TabletEvent::Up).unwrap();
        let mut out = ctx.run_ui(raw(1.1), |ui| app.ui_inner(ui));
        out.textures_delta.clear();

        // Dans la fenêtre : x=0.5 -> 1280 px écran -> UI x = 280 ;
        // y=0.3 -> 432 px écran -> UI y = 232 (dans le canvas central).
        tx.send(TabletEvent::Down {
            x: 0.5,
            y: 0.3,
            pressure: 1.0,
        })
        .unwrap();
        let mut out = ctx.run_ui(raw(1.2), |ui| app.ui_inner(ui));
        out.textures_delta.clear();
        assert!(
            app.current.is_some(),
            "dans la fenêtre : le stroke doit s'ouvrir"
        );
        tx.send(TabletEvent::Move {
            x: 0.55,
            y: 0.3,
            pressure: 1.0,
        })
        .unwrap();
        let mut out = ctx.run_ui(raw(1.3), |ui| app.ui_inner(ui));
        out.textures_delta.clear();
        tx.send(TabletEvent::Up).unwrap();
        let mut out = ctx.run_ui(raw(1.4), |ui| app.ui_inner(ui));
        out.textures_delta.clear();
        assert_eq!(app.strokes.len(), 1);
        let pts = &app.strokes[0].points;
        assert!(pts.len() >= 2, "points={}", pts.len());
        // Coordonnées relatives au canvas (rect.min = (8, 34) dans ce layout) :
        // x attendu = 280 - 8 = 272, y attendu = 232 - 34 = 198.
        assert!(
            (pts[0].x - 272.0).abs() < 1.0,
            "x canvas attendu 272, eu {}",
            pts[0].x
        );
        assert!(
            (pts[0].y - 198.0).abs() < 1.0,
            "y canvas attendu 198, eu {}",
            pts[0].y
        );
    }

    // ---- Record / replay d'input (--record-input / --replay-input) ----

    fn raw_frame(t: f64, events: Vec<egui::Event>) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            time: Some(t),
            events,
            ..Default::default()
        }
    }

    fn tmp_record_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ecriture-rs-{tag}-{}.jsonl", std::process::id()))
    }

    /// Un scénario souris enregistré via ui_inner (clic-maintient-relâche)
    /// doit produire exactement le même stroke en replay headless.
    #[test]
    fn record_puis_replay_reproduit_les_strokes_souris() {
        // Sanity : un RawInput avec des événements se sérialise bien en JSONL.
        let raw = raw_frame(
            1.0,
            vec![egui::Event::PointerMoved(egui::pos2(10.0, 20.0))],
        );
        let line = serde_json::to_string(&RecordedLine::Frame(raw)).expect("serialize RawInput");
        let back: RecordedLine = serde_json::from_str(&line).expect("deserialize RawInput");
        match back {
            RecordedLine::Frame(r) => assert_eq!(r.events.len(), 1),
            _ => panic!("mauvaise variante"),
        }

        let ctx = egui::Context::default();
        let path = tmp_record_path("souris");
        let _ = std::fs::remove_file(&path);
        let mut app = app_test();
        app.recorder = Some(std::io::BufWriter::new(std::fs::File::create(&path).unwrap()));
        for (t, evs) in [
            (
                1.0,
                vec![
                    egui::Event::PointerMoved(egui::pos2(400.0, 300.0)),
                    ptr_button(egui::pos2(400.0, 300.0), true),
                ],
            ),
            (1.05, vec![egui::Event::PointerMoved(egui::pos2(450.0, 320.0))]),
            (
                1.1,
                vec![
                    egui::Event::PointerMoved(egui::pos2(460.0, 330.0)),
                    ptr_button(egui::pos2(460.0, 330.0), false),
                ],
            ),
        ] {
            let mut out = ctx.run_ui(raw_frame(t, evs), |ui| app.ui_inner(ui));
            out.textures_delta.clear();
        }
        app.flush_recorder();
        drop(app);
        let (app2, n) = replay_file(path.to_str().unwrap(), &Config::default()).unwrap();
        assert_eq!(n, 3, "3 frames enregistrées");
        assert_eq!(app2.strokes.len(), 1, "un stroke archivé après le clic");
        assert!(app2.strokes[0].points.len() >= 2);
        assert!(app2.current.is_none());
        let _ = std::fs::remove_file(&path);
    }

    /// Écrit un enregistrement de démo (clic-maintient-relâche) dans
    /// /tmp/ecriture-rs-demo-input.jsonl pour tester le CLI :
    /// `cargo test --bin ecriture-rs -- --ignored record_input_demo_dans_tmp`
    /// puis `cargo run -- --replay-input /tmp/ecriture-rs-demo-input.jsonl`
    #[test]
    #[ignore]
    fn record_input_demo_dans_tmp() {
        let ctx = egui::Context::default();
        let path = std::path::PathBuf::from("/tmp/ecriture-rs-demo-input.jsonl");
        let mut app = app_test();
        app.recorder = Some(std::io::BufWriter::new(std::fs::File::create(&path).unwrap()));
        for (t, evs) in [
            (
                1.0,
                vec![
                    egui::Event::PointerMoved(egui::pos2(400.0, 300.0)),
                    ptr_button(egui::pos2(400.0, 300.0), true),
                ],
            ),
            (1.05, vec![egui::Event::PointerMoved(egui::pos2(450.0, 320.0))]),
            (
                1.1,
                vec![
                    egui::Event::PointerMoved(egui::pos2(460.0, 330.0)),
                    ptr_button(egui::pos2(460.0, 330.0), false),
                ],
            ),
        ] {
            let mut out = ctx.run_ui(raw_frame(t, evs), |ui| app.ui_inner(ui));
            out.textures_delta.clear();
        }
        app.flush_recorder();
        eprintln!("record demo -> {}", path.display());
    }

    /// Les événements tablette evdev (qui ne passent pas par winit) doivent
    /// aussi être enregistrés (lignes Tablet) et rejoués via le channel.
    #[test]
    fn record_puis_replay_reproduit_les_strokes_tablette() {
        let ctx = egui::Context::default();
        let path = tmp_record_path("tablette");
        let _ = std::fs::remove_file(&path);
        let (tx, rx) = channel();
        let src = ModelSource::Repo("Sachin-0001/crnn-iam-handwriting".into());
        let mut app = InkApp::with_tablet_rx(
            Box::new(MockRecognizer),
            "mock".to_string(),
            model_choices(&src),
            0,
            Config::default(),
            DecodeOptions::default(),
            Some(rx),
        );
        app.recorder = Some(std::io::BufWriter::new(std::fs::File::create(&path).unwrap()));
        tx.send(TabletEvent::Down {
            x: 0.5,
            y: 0.5,
            pressure: 1.0,
        })
        .unwrap();
        let mut out = ctx.run_ui(raw_frame(1.0, vec![]), |ui| app.ui_inner(ui));
        out.textures_delta.clear();
        tx.send(TabletEvent::Move {
            x: 0.6,
            y: 0.5,
            pressure: 1.0,
        })
        .unwrap();
        let mut out = ctx.run_ui(raw_frame(1.1, vec![]), |ui| app.ui_inner(ui));
        out.textures_delta.clear();
        tx.send(TabletEvent::Up).unwrap();
        let mut out = ctx.run_ui(raw_frame(1.2, vec![]), |ui| app.ui_inner(ui));
        out.textures_delta.clear();
        app.flush_recorder();
        drop(app);
        let (app2, n) = replay_file(path.to_str().unwrap(), &Config::default()).unwrap();
        assert_eq!(n, 6, "3 frames egui + 3 batches tablette");
        assert_eq!(app2.strokes.len(), 1, "le stroke stylet doit être archivé");
        assert!(
            app2.strokes[0].points.len() >= 2,
            "points={}",
            app2.strokes[0].points.len()
        );
        let _ = std::fs::remove_file(&path);
    }
}
