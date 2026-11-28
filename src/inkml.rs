//! Lecture de fichiers **InkML** (W3C) — le sous-ensemble utile pour les bases
//! IAM / CROHME / nos propres exports (`ink::strokes_to_inkml`) :
//!
//! - `<trace id="N">x y [t], x y [t], …</trace>` (2 ou 3 colonnes par point ;
//!   temps synthétique si absent : +20 ms/point, +200 ms/stroke),
//! - `<annotation type="truth">…</annotation>` → vérité terrain,
//! - `<traceGroup><traceGroup>…<traceView traceDataRef="N"/>…</traceGroup>…`
//!   → segmentation en mots/symboles (groupes d'indices de strokes).
//!
//! Porté d'après `hwrt/datasets/inkml.py` (MartinThoma/hwrt, licence MIT) :
//! traces triées par `id`, première annotation `truth` retenue.
//!
//! Parseur XML minimal fait main (pas de nouvelle dépendance) : gère les
//! namespaces, commentaires, CDATA, PI et les entités usuelles.

use std::path::Path;

use anyhow::{Context, Result};

use crate::ink::{Point, Stroke};

/// Un échantillon InkML : strokes + vérité terrain + segmentation en groupes.
#[derive(Clone, Debug)]
pub struct InkmlSample {
    pub strokes: Vec<Stroke>,
    pub truth: Option<String>,
    /// Groupes de strokes (mots/symboles) si des traceGroup imbriqués existent :
    /// chaque groupe = indices dans `strokes`.
    pub groups: Vec<Vec<usize>>,
}

enum XmlEvent {
    Start {
        name: String,
        attrs: Vec<(String, String)>,
    },
    Text(String),
    End {
        name: String,
    },
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find('&') {
        out.push_str(&rest[..pos]);
        let semi = rest[pos..].find(';').map(|i| pos + i);
        match semi {
            Some(end) => {
                let ent = &rest[pos + 1..end];
                let rep = match ent {
                    "lt" => "<",
                    "gt" => ">",
                    "amp" => "&",
                    "quot" => "\"",
                    "apos" => "'",
                    _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                        let Ok(cp) = u32::from_str_radix(&ent[2..], 16) else {
                            out.push_str(&rest[pos..=end]);
                            rest = &rest[end + 1..];
                            continue;
                        };
                        out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        rest = &rest[end + 1..];
                        continue;
                    }
                    _ if ent.starts_with('#') => {
                        let Ok(cp) = ent[1..].parse::<u32>() else {
                            out.push_str(&rest[pos..=end]);
                            rest = &rest[end + 1..];
                            continue;
                        };
                        out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        rest = &rest[end + 1..];
                        continue;
                    }
                    _ => {
                        out.push_str(&rest[pos..=end]);
                        rest = &rest[end + 1..];
                        continue;
                    }
                };
                out.push_str(rep);
                rest = &rest[end + 1..];
            }
            None => {
                out.push_str(&rest[pos..]);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

/// Nom local d'un élément XML (sans préfixe de namespace).
fn local_name(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

fn parse_events(s: &str) -> Result<Vec<XmlEvent>> {
    let mut evts = Vec::new();
    let mut text = String::new();
    let mut i = 0usize;
    let flush = |text: &mut String, evts: &mut Vec<XmlEvent>| {
        if !text.is_empty() {
            evts.push(XmlEvent::Text(std::mem::take(text)));
        }
    };
    while i < s.len() {
        if s[i..].starts_with('<') {
            if s[i..].starts_with("<!--") {
                let end = s[i..]
                    .find("-->")
                    .context("commentaire XML non fermé")?
                    + i
                    + 3;
                i = end;
                continue;
            }
            if s[i..].starts_with("<![CDATA[") {
                let end = s[i..].find("]]>").context("CDATA non fermé")? + i + 3;
                text.push_str(&s[i + 9..end - 3]);
                i = end;
                continue;
            }
            if s[i..].starts_with("<?") {
                let end = s[i..].find("?>").context("PI non fermée")? + i + 2;
                i = end;
                continue;
            }
            flush(&mut text, &mut evts);
            if s[i..].starts_with("</") {
                let end = s[i..].find('>').context("balise fermante mal formée")? + i + 1;
                let name = s[i + 2..end - 1].trim().to_string();
                evts.push(XmlEvent::End { name });
                i = end;
                continue;
            }
            // balise ouvrante
            let end = s[i..].find('>').context("balise ouvrante mal formée")? + i + 1;
            let inner = &s[i + 1..end - 1];
            let self_closing = inner.trim_end().ends_with('/');
            let inner = if self_closing {
                inner.trim_end().trim_end_matches('/')
            } else {
                inner
            };
            let mut parts = inner.splitn(2, char::is_whitespace);
            let name = parts.next().unwrap_or("").to_string();
            let mut attrs = Vec::new();
            let rest = parts.next().unwrap_or("");
            let mut j = 0usize;
            let rb = rest.as_bytes();
            while j < rest.len() {
                while j < rest.len() && (rb[j] as char).is_whitespace() {
                    j += 1;
                }
                if j >= rest.len() {
                    break;
                }
                let eq = rest[j..].find('=').map(|e| j + e).context("attribut sans =")?;
                let aname = rest[j..eq].trim().to_string();
                let mut k = eq + 1;
                while k < rest.len() && (rest.as_bytes()[k] as char).is_whitespace() {
                    k += 1;
                }
                let quote = rest.as_bytes()[k];
                assert!(quote == b'"' || quote == b'\'', "attribut sans guillemets");
                let close = rest[k + 1..]
                    .find(quote as char)
                    .map(|c| k + 1 + c)
                    .context("guillemet non fermé")?;
                attrs.push((aname, unescape(&rest[k + 1..close])));
                j = close + 1;
            }
            evts.push(XmlEvent::Start { name, attrs });
            if self_closing {
                evts.push(XmlEvent::End {
                    name: evts_name_last(&evts),
                });
            }
            i = end;
        } else {
            let next = s[i..].find('<').map(|p| i + p).unwrap_or(s.len());
            text.push_str(&s[i..next]);
            i = next;
        }
    }
    flush(&mut text, &mut evts);
    Ok(evts)
}

fn evts_name_last(evts: &[XmlEvent]) -> String {
    match evts.last() {
        Some(XmlEvent::Start { name, .. }) => name.clone(),
        _ => String::new(),
    }
}

/// Parse une chaîne InkML en échantillon (strokes + truth + groupes).
pub fn sample_from_str(s: &str) -> Result<InkmlSample> {
    let evts = parse_events(s)?;
    let mut traces: Vec<(u32, String)> = Vec::new(); // (id, texte brut)
    let mut truth: Option<String> = None;
    let mut groups: Vec<Vec<usize>> = Vec::new();

    let mut cur_trace_id: Option<u32> = None;
    let mut cur_trace_text = String::new();
    let mut in_truth = false;
    let mut truth_text = String::new();
    let mut tg_depth = 0usize;

    for evt in &evts {
        match evt {
            XmlEvent::Start { name, attrs } => {
                let local = local_name(name);
                match local {
                    "trace" => {
                        let id = attrs
                            .iter()
                            .find(|(k, _)| k == "id")
                            .and_then(|(_, v)| v.parse::<u32>().ok())
                            .unwrap_or(traces.len() as u32);
                        cur_trace_id = Some(id);
                        cur_trace_text.clear();
                    }
                    "annotation" => {
                        let ty = attrs
                            .iter()
                            .find(|(k, _)| k == "type")
                            .map(|(_, v)| v.as_str());
                        if ty == Some("truth") {
                            in_truth = true;
                            truth_text.clear();
                        }
                    }
                    "traceGroup" => {
                        tg_depth += 1;
                        if tg_depth == 2 {
                            groups.push(Vec::new());
                        }
                    }
                    "traceView" => {
                        if tg_depth >= 2 {
                            if let Some((_, v)) = attrs.iter().find(|(k, _)| k == "traceDataRef") {
                                if let Ok(id) = v.parse::<u32>() {
                                    if let Some(g) = groups.last_mut() {
                                        g.push(id as usize);
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            XmlEvent::Text(t) => {
                if cur_trace_id.is_some() {
                    cur_trace_text.push_str(t);
                }
                if in_truth {
                    truth_text.push_str(t);
                }
            }
            XmlEvent::End { name } => {
                let local = local_name(name);
                match local {
                    "trace" => {
                        if let Some(id) = cur_trace_id.take() {
                            traces.push((id, std::mem::take(&mut cur_trace_text)));
                        }
                    }
                    "annotation" => {
                        if in_truth {
                            in_truth = false;
                            let t = unescape(truth_text.trim());
                            if truth.is_none() && !t.is_empty() {
                                truth = Some(t);
                            }
                        }
                    }
                    "traceGroup" => {
                        tg_depth = tg_depth.saturating_sub(1);
                    }
                    _ => {}
                }
            }
        }
    }

    // traces triées par id (comme hwrt), index id -> position
    traces.sort_by_key(|(id, _)| *id);
    let id_to_idx: std::collections::HashMap<usize, usize> = traces
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (*id as usize, i))
        .collect();

    // points : 2 ou 3 colonnes ; temps synthétique si absent (+20 ms/point, +200 ms/stroke)
    let mut strokes = Vec::with_capacity(traces.len());
    let mut time = 0.0f64;
    for (_, text) in &traces {
        let mut st = Stroke::new();
        for pt in text.split(',') {
            let vals: Vec<&str> = pt.split_whitespace().collect();
            if vals.len() < 2 {
                continue;
            }
            let x: f32 = vals[0].parse().context("x non numérique")?;
            let y: f32 = vals[1].parse().context("y non numérique")?;
            let t = if vals.len() >= 3 {
                vals[2].parse::<f64>().context("time non numérique")?
            } else {
                let t = time;
                time += 20.0;
                t
            };
            st.push(Point {
                x,
                y,
                t,
                pressure: 1.0,
            });
        }
        time += 200.0;
        strokes.push(st);
    }

    // groupes : ids de traces -> indices de strokes
    let groups: Vec<Vec<usize>> = groups
        .into_iter()
        .map(|g| {
            g.into_iter()
                .filter_map(|id| id_to_idx.get(&id).copied())
                .collect()
        })
        .filter(|g: &Vec<usize>| !g.is_empty())
        .collect();

    Ok(InkmlSample {
        strokes,
        truth,
        groups,
    })
}

/// Lit un fichier InkML.
pub fn read_inkml(path: &Path) -> Result<InkmlSample> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("lecture de {}", path.display()))?;
    sample_from_str(&s).with_context(|| format!("parse de {}", path.display()))
}

/// Lit tous les `*.inkml` d'un dossier (triés par nom) ; retourne (nom, échantillon).
pub fn read_inkml_dir(dir: &Path) -> Result<Vec<(String, InkmlSample)>> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("lecture du dossier {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "inkml").unwrap_or(false))
        .collect();
    files.sort();
    let mut out = Vec::new();
    for f in files {
        let name = f
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        out.push((name, read_inkml(&f)?));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI: &str = r##"<?xml version="1.0"?>
<ink xmlns="http://www.w3.org/2003/InkML">
  <annotation type="truth">a &lt; b &amp; c</annotation>
  <annotation type="writer">007</annotation>
  <trace id="1">30 40 100.5, 35 42 101.0</trace>
  <trace id="0">10 10, 20 20, 30 10</trace>
  <traceGroup>
    <traceGroup>
      <annotation type="segment">a</annotation>
      <traceView traceDataRef="0"/>
    </traceGroup>
    <traceGroup>
      <annotation type="segment">&lt;b</annotation>
      <traceView traceDataRef="1"/>
    </traceGroup>
  </traceGroup>
</ink>
"##;

    #[test]
    fn parse_mini_inkml() {
        let s = sample_from_str(MINI).unwrap();
        // 2 traces, triées par id : id 0 d'abord
        assert_eq!(s.strokes.len(), 2);
        assert_eq!(s.strokes[0].points.len(), 3);
        assert_eq!(s.strokes[1].points.len(), 2);
        // trace id=1 a des temps explicites
        assert!((s.strokes[1].points[0].t - 100.5).abs() < 1e-9);
        // trace id=0 : temps synthétique (+20/point, +200/stroke)
        assert!((s.strokes[0].points[0].t - 0.0).abs() < 1e-9);
        assert!((s.strokes[0].points[1].t - 20.0).abs() < 1e-9);
        assert!((s.strokes[0].points[2].t - 40.0).abs() < 1e-9);
        // vérité terrain avec entités déséchappées
        assert_eq!(s.truth.as_deref(), Some("a < b & c"));
        // groupes de strokes (mots) : ids -> indices triés par id
        assert_eq!(s.groups, vec![vec![0], vec![1]]);
    }

    #[test]
    fn unescape_entites() {
        assert_eq!(unescape("a &lt; b &amp; &quot;c&quot; &#65; &#x42;"), "a < b & \"c\" A B");
    }

    #[test]
    fn cdata_et_commentaires() {
        let s = r##"<ink><!-- commentaire --><trace id="0"><![CDATA[1 2, 3 4]]></trace><annotation type="truth"><![CDATA[hi]]></annotation></ink>"##;
        let s = sample_from_str(s).unwrap();
        assert_eq!(s.strokes.len(), 1);
        assert_eq!(s.strokes[0].points.len(), 2);
        assert_eq!(s.truth.as_deref(), Some("hi"));
    }
}
