//! Pilote de périphériques virtuels (souris / stylet) via `/dev/uinput`.
//!
//! Sert à tester l'interface d'ecriture-rs de bout en bout : le device virtuel
//! apparaît dans la session (Wayland/X11) comme un vrai périphérique, on peut
//! donc injecter des scénarios réalistes (clic-maintient-relâche, hover,
//! contact stylet avec/sans Up, etc.) pendant que l'app tourne.
//!
//! Nécessite l'accès en écriture à /dev/uinput (groupe `input` ou `uinput`) :
//! `sudo usermod -aG input $USER` puis re-login.
//!
//! Usage :
//!   cargo run --release --bin uinput-pilot -- tools/scenario.txt
//!
//! Format du scénario (une commande par ligne, `#` = commentaire) :
//!   screen W H      résolution de l'écran en pixels (défaut 1920x1080),
//!                   utilisée pour les coordonnées absolues du stylet
//!   mouse           crée la souris virtuelle (REL_X/REL_Y + BTN_LEFT/RIGHT/MIDDLE)
//!   pen             crée le stylet virtuel (ABS_X/ABS_Y/ABS_PRESSURE + BTN_TOUCH
//!                   + BTN_TOOL_PEN, comme un vrai stylet Wacom)
//!   mmove dx dy     mouvement relatif de la souris
//!   mdown / mup     bouton gauche enfoncé / relâché
//!   mclick          clic complet (down + up)
//!   pmove x y       déplace le stylet aux coordonnées écran (pixels, origine
//!                   en haut à gauche de l'écran `screen W H`)
//!   pdown [p]       contact du stylet, pression p dans 0..1 (défaut 0.8)
//!   pup             relève le stylet (fin du contact)
//!   sleep ms        attend ms millisecondes

use std::thread::sleep;
use std::time::Duration;

use evdev::uinput::VirtualDevice;
use evdev::{
    AbsInfo, AbsoluteAxisCode, AttributeSet, BusType, EventType, InputEvent, InputId, KeyCode,
    RelativeAxisCode, SynchronizationCode, UinputAbsSetup,
};

/// Pression max de l'axe ABS_PRESSURE du stylet virtuel (comme un Wacom).
const PRESSURE_MAX: i32 = 1023;

/// Une commande d'un scénario d'injection.
#[derive(Debug, Clone, PartialEq)]
enum Cmd {
    Screen(i32, i32),
    Mouse,
    Pen,
    MMove(i32, i32),
    MDown,
    MUp,
    MClick,
    PMove(i32, i32),
    PDown(f32),
    PUp,
    Sleep(u64),
}

/// Parse un scénario texte en commandes. Erreur avec numéro de ligne sinon.
fn parse_scenario(text: &str) -> Result<Vec<Cmd>, String> {
    let mut cmds = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        let err = |m: &str| format!("ligne {}: {m} (`{line}`)", i + 1);
        let cmd = match words[0] {
            "screen" => {
                let w = words
                    .get(1)
                    .and_then(|v| v.parse().ok())
                    .filter(|v| *v > 0)
                    .ok_or_else(|| err("screen attend W H (pixels)"))?;
                let h = words
                    .get(2)
                    .and_then(|v| v.parse().ok())
                    .filter(|v| *v > 0)
                    .ok_or_else(|| err("screen attend W H (pixels)"))?;
                Cmd::Screen(w, h)
            }
            "mouse" => Cmd::Mouse,
            "pen" => Cmd::Pen,
            "mmove" => {
                let dx = words
                    .get(1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| err("mmove attend dx dy"))?;
                let dy = words
                    .get(2)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| err("mmove attend dx dy"))?;
                Cmd::MMove(dx, dy)
            }
            "mdown" => Cmd::MDown,
            "mup" => Cmd::MUp,
            "mclick" => Cmd::MClick,
            "pmove" => {
                let x = words
                    .get(1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| err("pmove attend x y (pixels écran)"))?;
                let y = words
                    .get(2)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| err("pmove attend x y (pixels écran)"))?;
                Cmd::PMove(x, y)
            }
            "pdown" => {
                let p = words
                    .get(1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.8f32)
                    .clamp(0.0, 1.0);
                Cmd::PDown(p)
            }
            "pup" => Cmd::PUp,
            "sleep" => {
                let ms = words
                    .get(1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| err("sleep attend ms"))?;
                Cmd::Sleep(ms)
            }
            other => return Err(err(&format!("commande inconnue: {other}"))),
        };
        cmds.push(cmd);
    }
    Ok(cmds)
}

/// Événement SYN_REPORT (fin d'un paquet atomique d'événements).
fn syn() -> InputEvent {
    InputEvent::new(
        EventType::SYNCHRONIZATION.0,
        SynchronizationCode::SYN_REPORT.0,
        0,
    )
}

fn key(code: KeyCode, value: i32) -> InputEvent {
    InputEvent::new(EventType::KEY.0, code.0, value)
}

fn rel(code: RelativeAxisCode, value: i32) -> InputEvent {
    InputEvent::new(EventType::RELATIVE.0, code.0, value)
}

fn abs(code: AbsoluteAxisCode, value: i32) -> InputEvent {
    InputEvent::new(EventType::ABSOLUTE.0, code.0, value)
}

/// Émet un paquet d'événements + SYN_REPORT, puis laisse au compositor le
/// temps de traiter (15 ms).
fn emit(dev: &mut VirtualDevice, mut evs: Vec<InputEvent>) -> std::io::Result<()> {
    evs.push(syn());
    dev.emit(&evs)?;
    sleep(Duration::from_millis(15));
    Ok(())
}

/// Le pilote : détient les devices virtuels créés et exécute les commandes.
struct Pilot {
    mouse: Option<VirtualDevice>,
    pen: Option<VirtualDevice>,
    screen: (i32, i32),
    pen_pos: (i32, i32),
    pen_down: bool,
    pen_pressure: f32,
}

impl Pilot {
    fn new() -> Self {
        Self {
            mouse: None,
            pen: None,
            screen: (1920, 1080),
            pen_pos: (0, 0),
            pen_down: false,
            pen_pressure: 0.0,
        }
    }

    /// Crée la souris virtuelle.
    fn create_mouse(&mut self) -> std::io::Result<()> {
        let keys = AttributeSet::from_iter([
            KeyCode::BTN_LEFT,
            KeyCode::BTN_RIGHT,
            KeyCode::BTN_MIDDLE,
        ]);
        let rels = AttributeSet::from_iter([RelativeAxisCode::REL_X, RelativeAxisCode::REL_Y]);
        let dev = VirtualDevice::builder()?
            .name("vibe-pilot-mouse")
            .input_id(InputId::new(BusType::BUS_USB, 0x16c0, 0x05df, 1))
            .with_keys(&keys)?
            .with_relative_axes(&rels)?
            .build()?;
        eprintln!("uinput-pilot: souris virtuelle 'vibe-pilot-mouse' créée");
        self.mouse = Some(dev);
        Ok(())
    }

    /// Crée le stylet virtuel (ABS_X/Y bornés à l'écran, ABS_PRESSURE, BTN_TOUCH
    /// + BTN_TOOL_PEN) — même profil qu'un stylet Wacom, donc détecté comme
    /// tablette par ecriture-rs (is_tablet).
    fn create_pen(&mut self) -> std::io::Result<()> {
        let (w, h) = self.screen;
        let abs_x = UinputAbsSetup::new(
            AbsoluteAxisCode::ABS_X,
            AbsInfo::new(0, 0, w - 1, 0, 0, 0),
        );
        let abs_y = UinputAbsSetup::new(
            AbsoluteAxisCode::ABS_Y,
            AbsInfo::new(0, 0, h - 1, 0, 0, 0),
        );
        let abs_p = UinputAbsSetup::new(
            AbsoluteAxisCode::ABS_PRESSURE,
            AbsInfo::new(0, 0, PRESSURE_MAX, 0, 0, 0),
        );
        let keys = AttributeSet::from_iter([KeyCode::BTN_TOUCH, KeyCode::BTN_TOOL_PEN]);
        let dev = VirtualDevice::builder()?
            .name("vibe-pilot-pen")
            .input_id(InputId::new(BusType::BUS_USB, 0x16c0, 0x05df, 2))
            .with_absolute_axis(&abs_x)?
            .with_absolute_axis(&abs_y)?
            .with_absolute_axis(&abs_p)?
            .with_keys(&keys)?
            .build()?;
        eprintln!("uinput-pilot: stylet virtuel 'vibe-pilot-pen' créé");
        self.pen = Some(dev);
        Ok(())
    }

    fn exec(&mut self, cmd: Cmd) -> Result<(), String> {
        match cmd {
            Cmd::Screen(w, h) => {
                self.screen = (w, h);
                // Le stylet doit être recréé pour prendre en compte la nouvelle
                // résolution (bornes ABS_X/ABS_Y).
                if self.pen.is_some() {
                    eprintln!("uinput-pilot: `screen` après `pen` -> recrée le stylet");
                    self.pen = None;
                    self.create_pen().map_err(|e| e.to_string())?;
                }
            }
            Cmd::Mouse => self.create_mouse().map_err(|e| e.to_string())?,
            Cmd::Pen => self.create_pen().map_err(|e| e.to_string())?,
            Cmd::MMove(dx, dy) => {
                let dev = self.mouse.as_mut().ok_or("pas de souris virtuelle (`mouse`)")?;
                emit(
                    dev,
                    vec![
                        rel(RelativeAxisCode::REL_X, dx),
                        rel(RelativeAxisCode::REL_Y, dy),
                    ],
                )
                .map_err(|e| e.to_string())?;
            }
            Cmd::MDown => {
                let dev = self.mouse.as_mut().ok_or("pas de souris virtuelle (`mouse`)")?;
                emit(dev, vec![key(KeyCode::BTN_LEFT, 1)]).map_err(|e| e.to_string())?;
            }
            Cmd::MUp => {
                let dev = self.mouse.as_mut().ok_or("pas de souris virtuelle (`mouse`)")?;
                emit(dev, vec![key(KeyCode::BTN_LEFT, 0)]).map_err(|e| e.to_string())?;
            }
            Cmd::MClick => {
                let dev = self.mouse.as_mut().ok_or("pas de souris virtuelle (`mouse`)")?;
                emit(dev, vec![key(KeyCode::BTN_LEFT, 1)]).map_err(|e| e.to_string())?;
                sleep(Duration::from_millis(30));
                emit(dev, vec![key(KeyCode::BTN_LEFT, 0)]).map_err(|e| e.to_string())?;
            }
            Cmd::PMove(x, y) => {
                let (w, h) = self.screen;
                let x = x.clamp(0, w - 1);
                let y = y.clamp(0, h - 1);
                self.pen_pos = (x, y);
                let dev = self.pen.as_mut().ok_or("pas de stylet virtuel (`pen`)")?;
                let mut evs = vec![
                    abs(AbsoluteAxisCode::ABS_X, x),
                    abs(AbsoluteAxisCode::ABS_Y, y),
                ];
                if self.pen_down {
                    evs.push(abs(
                        AbsoluteAxisCode::ABS_PRESSURE,
                        (self.pen_pressure * PRESSURE_MAX as f32) as i32,
                    ));
                }
                emit(dev, evs).map_err(|e| e.to_string())?;
            }
            Cmd::PDown(p) => {
                self.pen_down = true;
                self.pen_pressure = p;
                let dev = self.pen.as_mut().ok_or("pas de stylet virtuel (`pen`)")?;
                let (x, y) = self.pen_pos;
                emit(
                    dev,
                    vec![
                        key(KeyCode::BTN_TOOL_PEN, 1),
                        key(KeyCode::BTN_TOUCH, 1),
                        abs(AbsoluteAxisCode::ABS_X, x),
                        abs(AbsoluteAxisCode::ABS_Y, y),
                        abs(
                            AbsoluteAxisCode::ABS_PRESSURE,
                            (p * PRESSURE_MAX as f32) as i32,
                        ),
                    ],
                )
                .map_err(|e| e.to_string())?;
            }
            Cmd::PUp => {
                self.pen_down = false;
                let dev = self.pen.as_mut().ok_or("pas de stylet virtuel (`pen`)")?;
                emit(
                    dev,
                    vec![
                        abs(AbsoluteAxisCode::ABS_PRESSURE, 0),
                        key(KeyCode::BTN_TOUCH, 0),
                        key(KeyCode::BTN_TOOL_PEN, 0),
                    ],
                )
                .map_err(|e| e.to_string())?;
            }
            Cmd::Sleep(ms) => sleep(Duration::from_millis(ms)),
        }
        Ok(())
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: uinput-pilot FICHIER_SCENARIO");
        eprintln!("exemple: cargo run --release --bin uinput-pilot -- tools/scenario-stylet-puis-souris.txt");
        std::process::exit(2);
    }
    let text = match std::fs::read_to_string(&args[1]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("lecture de {}: {e}", args[1]);
            std::process::exit(1);
        }
    };
    let cmds = match parse_scenario(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("scénario invalide: {e}");
            std::process::exit(1);
        }
    };
    let mut pilot = Pilot::new();
    let n = cmds.len();
    for cmd in cmds {
        if let Err(e) = pilot.exec(cmd) {
            eprintln!("uinput-pilot: erreur: {e}");
            std::process::exit(1);
        }
    }
    eprintln!("uinput-pilot: scénario terminé ({n} commandes)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_scenario_commandes_connues() {
        let text = "\
# commentaire
screen 2560 1440
pen
pmove 100 200
pdown 0.5
pmove 300 400
pup
sleep 100
mouse
mmove 10 -5
mdown
mup
mclick
";
        let cmds = parse_scenario(text).unwrap();
        assert_eq!(
            cmds,
            vec![
                Cmd::Screen(2560, 1440),
                Cmd::Pen,
                Cmd::PMove(100, 200),
                Cmd::PDown(0.5),
                Cmd::PMove(300, 400),
                Cmd::PUp,
                Cmd::Sleep(100),
                Cmd::Mouse,
                Cmd::MMove(10, -5),
                Cmd::MDown,
                Cmd::MUp,
                Cmd::MClick,
            ]
        );
    }

    #[test]
    fn parse_scenario_pdown_pression_defaut_et_bornes() {
        assert_eq!(parse_scenario("pdown").unwrap(), vec![Cmd::PDown(0.8)]);
        assert_eq!(parse_scenario("pdown 2.0").unwrap(), vec![Cmd::PDown(1.0)]);
        assert_eq!(parse_scenario("pdown -1").unwrap(), vec![Cmd::PDown(0.0)]);
    }

    #[test]
    fn parse_scenario_erreurs_avec_numero_ligne() {
        assert!(parse_scenario("mmove 10").unwrap_err().contains("ligne 1"));
        assert!(parse_scenario("bogus").unwrap_err().contains("ligne 1"));
        assert!(parse_scenario("ok\nbogus").is_err());
        // "ok" n'existe pas -> ligne 1 ; testons une vraie commande valide d'abord
        assert!(parse_scenario("mouse\nbogus x").unwrap_err().contains("ligne 2"));
        assert!(parse_scenario("sleep abc").unwrap_err().contains("ligne 1"));
    }
}
