//! UI egui : canvas d'écriture + bouton de reconnaissance.

mod doctr;
mod ink;
mod lexicon;
mod model;
mod pylaia;
mod recognizer;
mod tablet;

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};

use anyhow::Context as _;
use eframe::egui;
use ink::{Point, Stroke};
use recognizer::{CrnnRecognizer, DecodeOptions, MockRecognizer, Norm, Recognizer};
use tablet::TabletEvent;

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
        Self::with_tablet_rx(
            recognizer,
            model_desc,
            model_sources,
            model_index,
            cfg,
            decode_opts,
            Some(rx),
        )
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

    /// Capture du tracé souris/touch dans le canvas. On ne se fie pas qu'à la
    /// Response du canvas : on lit l'état global du pointer (souris, ou stylet
    /// émulé en bouton principal) ET les événements touch bruts — un stylet
    /// sous Wayland arrive en `Event::Touch` avec une force (pression).
    ///
    /// Souris et touch sont traités SÉPARÉMENT : la souris ne trace que si un
    /// bouton est réellement enfoncé (`down`), donc un `touch_active` resté
    /// bloqué (Touch Start sans End) ne fait pas tracer la souris en hover.
    fn handle_pointer_input(&mut self, ui: &egui::Ui, rect: egui::Rect, now: f64) {
        let (mut touch_started, mut touch_ended, mut pressure, mut touch_pos) =
            (false, false, 1.0f32, None);
        ui.input(|i| {
            for ev in &i.events {
                if let egui::Event::Touch { phase, pos, force, .. } = ev {
                    match phase {
                        egui::TouchPhase::Start => {
                            touch_started = true;
                            touch_pos = Some(*pos);
                        }
                        egui::TouchPhase::Move => {
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
        if touch_started {
            self.touch_active = true;
        }
        if let Some(p) = touch_pos {
            self.last_touch_pos = Some(p);
        }
        if touch_ended {
            self.touch_active = false;
            self.last_touch_pos = None;
        }

        let (pressed, down, released, pointer_pos) = ui.input(|i| {
            let p = &i.pointer;
            (p.any_pressed(), p.any_down(), p.any_released(), p.latest_pos())
        });

        // Démarre un stroke uniquement sur un vrai début d'input (bouton
        // pressé ou touch start) avec le pointeur dans le canvas.
        let start_pos = touch_pos.or(pointer_pos);
        if (pressed || touch_started) && start_pos.map(|p| rect.contains(p)).unwrap_or(false) {
            self.current = Some(Stroke::new());
        }

        // Souris : tracer UNIQUEMENT si un bouton est réellement enfoncé.
        if down && !touch_ended {
            if let Some(p) = pointer_pos {
                if rect.contains(p) {
                    self.push_point(p, pressure, now, rect);
                }
            }
        }
        // Touch / stylet : tracer tant que le contact est actif.
        if self.touch_active && !touch_ended {
            if let Some(p) = touch_pos.or(self.last_touch_pos) {
                if rect.contains(p) {
                    self.push_point(p, pressure, now, rect);
                }
            }
        }
        // Termine le stroke sur relâchement / fin de touch, ou si plus aucun
        // input n'est actif (évite un stroke fantôme bloqué).
        if released || touch_ended || (!down && !self.touch_active) {
            self.end_stroke(now);
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
            // toujours le périphérique à egui). Coordonnées normalisées 0..1
            // de l'écran -> points UI via le screen_rect.
            let mut tablet_events = Vec::new();
            if let Some(rx) = &self.tablet_rx {
                while let Ok(ev) = rx.try_recv() {
                    tablet_events.push(ev);
                }
            }
            if !tablet_events.is_empty() {
                let screen = ui.ctx().viewport_rect();
                for ev in tablet_events {
                    self.tablet_count += 1;
                    match ev {
                        TabletEvent::Down { x, y, pressure } => {
                            let p = egui::pos2(x * screen.width(), y * screen.height());
                            if rect.contains(p) {
                                self.current = Some(Stroke::new());
                                self.push_point(p, pressure, now, rect);
                            }
                        }
                        TabletEvent::Move { x, y, pressure } => {
                            let p = egui::pos2(x * screen.width(), y * screen.height());
                            if rect.contains(p) {
                                self.push_point(p, pressure, now, rect);
                            }
                        }
                        TabletEvent::Up => {
                            self.end_stroke(now);
                        }
                    }
                }
            }

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
                        "down={} pressed={} released={} any_touches={} pos={:?}\n{}",
                        i.pointer.any_down(),
                        i.pointer.any_pressed(),
                        i.pointer.any_released(),
                        i.any_touches(),
                        i.pointer.latest_pos(),
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

/// Construit le reconnaisseur d'une source ; en cas d'échec, repli sur le
/// MockRecognizer et la description contient l'erreur.
fn build_recognizer(
    src: &ModelSource,
    cfg: &Config,
    opts: DecodeOptions,
    device: &candle_core::Device,
) -> (Box<dyn Recognizer>, String) {
    match try_build_recognizer(src, cfg, opts, device) {
        Ok((r, desc)) => (Box::new(r), desc),
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
         \x20         [--lexicon PATH|none] [--beam N] [--nbest N] [--lm-weight F]\n\
         défauts: --repo Sachin-0001/crnn-iam-handwriting (téléchargé via Hugging Face),\n\
         \x20         --input-h depuis config.json, --norm neg1, --lexicon embarqué (français),\n\
         \x20         --beam 16, --nbest 8, --lm-weight 2.0,\n\
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
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);
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
            "-h" | "--help" => usage_exit(),
            other => {
                eprintln!("argument inconnu: {other}");
                usage_exit();
            }
        }
    }
    cfg
}

fn main() -> eframe::Result<()> {
    let cfg = parse_args();

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
        "config: repo={} pth={:?} input_h={:?} input_w={:?} norm={:?} tablet_device={:?} beam={} nbest={} lm_weight={} lexique={}",
        cfg.repo,
        cfg.pth,
        cfg.input_h,
        cfg.input_w,
        cfg.norm,
        cfg.tablet_device,
        cfg.beam,
        cfg.nbest,
        cfg.lm_weight,
        lex_info,
    );
    let decode_opts = DecodeOptions {
        beam_size: cfg.beam,
        nbest: cfg.nbest,
        lm_weight: cfg.lm_weight,
        lexicon,
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

    let options = eframe::NativeOptions {
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
        // Pas de doublon si la source initiale est déjà un dépôt connu.
        let rimes = ModelSource::Repo("Teklia/pylaia-rimes".into());
        let v2 = model_choices(&rimes);
        assert_eq!(v2.iter().filter(|s| **s == rimes).count(), 1);
        assert_eq!(v2.len(), 3);
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
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            time: Some(1.0),
            events,
            ..Default::default()
        };
        let mut output = ctx.run_ui(raw, |ui| {
            let (response, _painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::drag());
            let now = ui.input(|i| i.time);
            app.handle_pointer_input(ui, response.rect, now);
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
}
