//! Lecture directe des périphériques stylet / dalle tactile via evdev
//! (`/dev/input/event*`). Sous Linux, winit peut ne remonter aucun
//! événement du stylet à egui ; on lit donc le périphérique nous-mêmes
//! dans un thread dédié et on injecte les points dans l'UI via un channel.
//!
//! Nécessite les droits de lecture sur /dev/input/event* (groupe `input`) :
//! `sudo usermod -aG input $USER` puis re-login.
//! Périphérique forcé : option CLI `--tablet-device /dev/input/eventN`,
//! ou variable d'env ECRITURE_TABLET_DEVICE=/dev/input/eventN.

use std::sync::mpsc::Sender;
use std::thread;

use evdev::{AbsoluteAxisCode, Device, EventType, KeyCode};

/// Événement stylet, coordonnées normalisées 0..1 dans l'espace de l'écran.
#[derive(Clone, Copy, Debug)]
pub enum TabletEvent {
    Down { x: f32, y: f32, pressure: f32 },
    Move { x: f32, y: f32, pressure: f32 },
    Up,
}

/// Lance un thread de lecture pour chaque périphérique tablette/stylet trouvé.
/// `device` force un périphérique précis ; sinon variable d'env
/// ECRITURE_TABLET_DEVICE ; sinon auto-détection sur /dev/input/event*.
pub fn spawn_readers(tx: Sender<TabletEvent>, device: Option<String>) {
    let paths = match device.or_else(|| std::env::var("ECRITURE_TABLET_DEVICE").ok()) {
        Some(p) => vec![p],
        None => list_event_devices(),
    };
    let mut spawned = 0;
    for path in &paths {
        match Device::open(path) {
            Ok(dev) if is_tablet(&dev) => {
                let name = dev.name().unwrap_or("?").to_string();
                eprintln!("tablette: lecture de {} ({})", path, name);
                let tx = tx.clone();
                let path = path.clone();
                thread::spawn(move || read_loop(path, tx));
                spawned += 1;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!(
                    "tablette: {path} inaccessible ({e}); ajoute l'utilisateur au groupe 'input'"
                );
            }
        }
    }
    if spawned == 0 {
        eprintln!("tablette: aucun périphérique stylet/touch trouvé sur /dev/input/event*");
    }
}

fn list_event_devices() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/dev/input") {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("event") {
                v.push(format!("/dev/input/{}", name));
            }
        }
    }
    v.sort();
    v
}

/// Un périphérique tablette : stylet (BTN_TOOL_PEN) ou dalle tactile (ABS_X/Y + BTN_TOUCH).
fn is_tablet(dev: &Device) -> bool {
    let has_pen = dev
        .supported_keys()
        .map(|k| k.contains(KeyCode::BTN_TOOL_PEN))
        .unwrap_or(false);
    let has_abs_xy = dev
        .supported_absolute_axes()
        .map(|a| {
            a.contains(AbsoluteAxisCode::ABS_X) && a.contains(AbsoluteAxisCode::ABS_Y)
        })
        .unwrap_or(false);
    let has_touch = dev
        .supported_keys()
        .map(|k| k.contains(KeyCode::BTN_TOUCH))
        .unwrap_or(false);
    has_pen || (has_abs_xy && has_touch)
}

fn read_loop(path: String, tx: Sender<TabletEvent>) {
    let mut dev = match Device::open(&path) {
        Ok(d) => d,
        Err(_) => return,
    };
    // Lecture bloquante : le thread dort entre les événements.
    let _ = dev.set_nonblocking(false);

    // Bornes des axes pour normaliser en 0..1.
    let absinfo: Vec<(AbsoluteAxisCode, i32)> = dev
        .get_absinfo()
        .map(|it| it.map(|(code, info)| (code, info.maximum())).collect())
        .unwrap_or_default();
    let max_of = |code: AbsoluteAxisCode| {
        absinfo
            .iter()
            .find(|(c, _)| *c == code)
            .map(|(_, m)| *m)
            .unwrap_or(1)
            .max(1) as f32
    };
    let max_x = max_of(AbsoluteAxisCode::ABS_X);
    let max_y = max_of(AbsoluteAxisCode::ABS_Y);
    let max_p = max_of(AbsoluteAxisCode::ABS_PRESSURE);

    // Contact réel : BTN_TOUCH (pointe ou doigt au contact). BTN_TOOL_PEN
    // passe à 1 dès la proximité (hover), avant le contact : on ne l'utilise
    // que si le périphérique n'a pas BTN_TOUCH.
    let contact_code = if dev
        .supported_keys()
        .map(|k| k.contains(KeyCode::BTN_TOUCH))
        .unwrap_or(false)
    {
        KeyCode::BTN_TOUCH.0
    } else {
        KeyCode::BTN_TOOL_PEN.0
    };

    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let mut pressure = 1.0f32;
    let mut down = false;
    let mut just_started = false;

    // fetch_events() fait un read() bloquant (fd en mode bloquant) puis
    // retourne les événements d'un batch ; il faut le rappeler en boucle,
    // sinon l'itérateur s'arrête au premier SYN_REPORT.
    loop {
        let batch = match dev.fetch_events() {
            Ok(b) => b,
            Err(_) => break, // périphérique débranché ou erreur de lecture
        };
        for ev in batch {
            let code = ev.code();
            let value = ev.value();
            match ev.event_type() {
                EventType::KEY => {
                    if code == contact_code {
                        if value == 1 && !down {
                            down = true;
                            just_started = true;
                        } else if value == 0 && down {
                            down = false;
                            let _ = tx.send(TabletEvent::Up);
                        }
                    }
                }
                EventType::ABSOLUTE => {
                    if code == AbsoluteAxisCode::ABS_X.0
                        || code == AbsoluteAxisCode::ABS_MT_POSITION_X.0
                    {
                        x = value as f32 / max_x;
                    } else if code == AbsoluteAxisCode::ABS_Y.0
                        || code == AbsoluteAxisCode::ABS_MT_POSITION_Y.0
                    {
                        y = value as f32 / max_y;
                    } else if code == AbsoluteAxisCode::ABS_PRESSURE.0 {
                        pressure = (value as f32 / max_p).clamp(0.0, 1.0);
                    }
                }
                // Fin d'un paquet d'événements atomique : on émet le point.
                // Le Down est émis ici (et pas au BTN_TOUCH) pour que les
                // coordonnées x/y soient à jour au moment du contact.
                EventType::SYNCHRONIZATION => {
                    if down {
                        if just_started {
                            just_started = false;
                            let _ = tx.send(TabletEvent::Down { x, y, pressure });
                        }
                        let _ = tx.send(TabletEvent::Move { x, y, pressure });
                    }
                }
                _ => {}
            }
        }
    }
    eprintln!("tablette: fin de lecture de {}", path);
}
