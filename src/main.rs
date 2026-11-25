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
}

impl InkApp {
    fn new(recognizer: Box<dyn Recognizer>, tablet_device: Option<String>, ctx: egui::Context) -> Self {
        // Lance la lecture evdev des périphériques stylet/touch en arrière-plan.
        // Le thread evdev réveille egui (request_repaint) à chaque événement.
        let (tx, rx) = channel();
        tablet::spawn_readers(tx, tablet_device, ctx);
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
            tablet_rx: Some(rx),
            tablet_count: 0,
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
            });
        });

        egui::Panel::bottom("result").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Texte reconnu :");
                // Texte finalisé en noir, hypothèse de la fenêtre en cours en gris.
                ui.label(egui::RichText::new(&self.result).monospace());
                if !self.pending_text.is_empty() {
                    ui.label(
                        egui::RichText::new(&self.pending_text)
                            .monospace()
                            .color(egui::Color32::GRAY),
                    );
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

            // Capture du tracé. On ne se fie pas qu'à la Response du canvas :
            // on lit l'état global du pointer (souris, ou stylet émulé en
            // bouton principal) ET les événements touch bruts — un stylet
            // sous Wayland arrive en Event::Touch avec une force (pression).
            let now = ui.input(|i| i.time);
            let rect = response.rect;

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
            let pos = pointer_pos.or(self.last_touch_pos);
            let pos_in = pos.filter(|p| rect.contains(*p));
            let drawing = down || self.touch_active;

            if (pressed || touch_started) && pos_in.is_some() {
                self.current = Some(Stroke::new());
            }
            if drawing && !touch_ended {
                if let Some(p) = pos {
                    if rect.contains(p) {
                        self.push_point(p, pressure, now, rect);
                    }
                }
            }
            if released || touch_ended {
                self.end_stroke(now);
            }

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
struct Config {
    /// Checkpoint .pth local ; vocab.json et config.json attendus à côté.
    pth: Option<String>,
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
         --repo Felix92/doctr-torch-crnn-mobilenet-v3-large-french : doctR FR téléchargé"
    );
    std::process::exit(2)
}

fn parse_args() -> Config {
    let mut cfg = Config {
        pth: None,
        repo: "Sachin-0001/crnn-iam-handwriting".to_string(),
        input_h: None,
        input_w: None,
        norm: Norm::NegOneOne,
        tablet_device: None,
        lexicon: None,
        beam: 16,
        nbest: 8,
        lm_weight: 2.0,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pth" => cfg.pth = Some(args.next().unwrap_or_else(|| usage_exit())),
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

    // Résout les fichiers du modèle : locaux (--pth + vocab.json/config.json
    // à côté) ou téléchargés depuis le dépôt HF (--repo).
    let (pth, vocab_json, config_json) = match &cfg.pth {
        Some(p) => {
            let p = PathBuf::from(p);
            let dir = p.parent().map(|d| d.to_path_buf()).unwrap_or_default();
            (p, dir.join("vocab.json"), dir.join("config.json"))
        }
        None => match model::download_model_files(&cfg.repo) {
            Ok(paths) => paths,
            Err(e) => {
                eprintln!("téléchargement HF impossible ({e:#}) : MockRecognizer");
                (
                    PathBuf::from("/nonexistent/model.pth"),
                    PathBuf::from("/nonexistent/vocab.json"),
                    PathBuf::from("/nonexistent/config.json"),
                )
            }
        },
    };
    let config_text = std::fs::read_to_string(&config_json).unwrap_or_default();
    let mconfig: model::ModelConfig =
        serde_json::from_str(&config_text).unwrap_or_default();
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
        "config: pth={} repo={} input_h={} input_w={} norm={:?} backend={} tablet_device={:?} beam={} nbest={} lm_weight={} lexique={}",
        pth.display(),
        cfg.repo,
        input_h,
        input_w,
        cfg.norm,
        if pylaia_syms.is_some() { "pylaia" } else if mconfig.is_doctr() { "doctr" } else { "sachin" },
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

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "ecriture-rs",
        options,
        Box::new(|cc| {
            let device = candle_core::Device::Cpu;
            // Modèle réel si présent, sinon bouchon pour tester l'UI.
            // Le format est détecté depuis les fichiers à côté du checkpoint :
            // syms.txt -> PyLaIA, arch/vocab dans config.json -> doctR,
            // sinon "Sachin-0001" (vocab.json + config.json).
            let built: anyhow::Result<CrnnRecognizer> = if let Some(syms) = &pylaia_syms {
                CrnnRecognizer::with_pylaia(&pth, syms, input_h, decode_opts, &device)
            } else if mconfig.is_doctr() {
                let dcfg: doctr::DoctrConfig = serde_json::from_str(&config_text)
                    .context("config.json doctR invalide")?;
                CrnnRecognizer::with_doctr(
                    &pth,
                    &dcfg,
                    128, // rnn_units doctR
                    input_h,
                    input_w,
                    decode_opts,
                    &device,
                )
            } else {
                CrnnRecognizer::with_options(
                    &pth,
                    &vocab_json,
                    mconfig.num_classes.unwrap_or(80),
                    mconfig.hidden_size.unwrap_or(256),
                    input_h,
                    cfg.norm,
                    decode_opts,
                    &device,
                )
            };
            let recognizer: Box<dyn Recognizer> = match built {
                Ok(r) => Box::new(r),
                Err(e) => {
                    eprintln!("{} illisible ({e:#}) : utilisation du MockRecognizer", pth.display());
                    Box::new(MockRecognizer)
                }
            };
            Ok(Box::new(InkApp::new(
                recognizer,
                cfg.tablet_device.clone(),
                cc.egui_ctx.clone(),
            )))
        }),
    )
}
