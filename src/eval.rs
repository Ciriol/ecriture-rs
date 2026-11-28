//! Métriques d'évaluation (CER/WER par distance de Levenshtein) et harnais de
//! benchmark : comparer la reconnaissance ligne entière vs par mots
//! (`wordseg::WordSegRecognizer`) sur un jeu d'échantillons (InkML IAM ou
//! images + GT), inspiré de l'évaluation hwrt (MartinThoma/hwrt, MIT).

use anyhow::Result;
use image::GrayImage;

use crate::ink::{self, Stroke};
use crate::recognizer::Recognizer;
use crate::wordseg::{self, SegParams};

/// Distance de Levenshtein entre deux séquences.
pub fn levenshtein<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// CER (Character Error Rate) = distance / nb de caractères de la référence.
pub fn cer(hyp: &str, gt: &str) -> f64 {
    let g: Vec<char> = gt.chars().collect();
    if g.is_empty() {
        return if hyp.is_empty() { 0.0 } else { 1.0 };
    }
    levenshtein(&hyp.chars().collect::<Vec<_>>(), &g) as f64 / g.len() as f64
}

/// WER (Word Error Rate) = distance / nb de mots de la référence.
pub fn wer(hyp: &str, gt: &str) -> f64 {
    let g: Vec<&str> = gt.split_whitespace().collect();
    if g.is_empty() {
        return if hyp.is_empty() { 0.0 } else { 1.0 };
    }
    levenshtein(&hyp.split_whitespace().collect::<Vec<_>>(), &g) as f64 / g.len() as f64
}

/// Un échantillon de benchmark : strokes (InkML) ou image, + vérité terrain.
pub struct BenchSample {
    pub name: String,
    pub gt: String,
    pub strokes: Option<Vec<Stroke>>,
    pub img: Option<GrayImage>,
    /// Nombre de mots de la GT (si segmentation connue), pour comparer au
    /// nombre de mots détectés.
    pub words_gt: Option<usize>,
}

impl BenchSample {
    pub fn from_inkml(name: String, s: &crate::inkml::InkmlSample) -> Option<Self> {
        let gt = s.truth.clone()?;
        Some(Self {
            name,
            gt,
            strokes: Some(s.strokes.clone()),
            img: None,
            words_gt: if s.groups.is_empty() {
                None
            } else {
                Some(s.groups.len())
            },
        })
    }

    pub fn from_image(name: String, img: GrayImage, gt: String) -> Self {
        let words_gt = Some(gt.split_whitespace().count());
        Self {
            name,
            gt,
            strokes: None,
            img: Some(img),
            words_gt,
        }
    }
}

/// Une ligne de résultats de benchmark.
pub struct BenchRow {
    pub name: String,
    pub gt: String,
    pub hyp_line: String,
    pub hyp_seg: String,
    pub cer_line: f64,
    pub cer_seg: f64,
    pub wer_line: f64,
    pub wer_seg: f64,
    pub words_det: Option<usize>,
    pub words_gt: Option<usize>,
}

/// Tourne les deux modes (ligne entière vs segmentation en mots) sur chaque
/// échantillon. `line` et `seg` doivent être deux instances du même modèle
/// (seg enveloppe le même reconnaisseur interne).
pub fn run_bench(
    samples: &[BenchSample],
    line: &mut dyn Recognizer,
    seg: &mut dyn Recognizer,
) -> Result<Vec<BenchRow>> {
    let mut rows = Vec::with_capacity(samples.len());
    for s in samples {
        let (hyp_line, words_det) = match &s.strokes {
            Some(st) => {
                let img = ink::render_to_image(st, 64);
                let nd = wordseg::detect_in_image(&img, &SegParams::default()).len();
                (line.recognize_strokes(st)?, Some(nd))
            }
            None => {
                let img = s.img.as_ref().expect("sample image");
                let nd = wordseg::detect_in_image(img, &SegParams::default()).len();
                (line.recognize(img)?, Some(nd))
            }
        };
        let hyp_seg = match &s.strokes {
            Some(st) => seg.recognize_strokes(st)?,
            None => seg.recognize(s.img.as_ref().expect("sample image"))?,
        };
        rows.push(BenchRow {
            name: s.name.clone(),
            gt: s.gt.clone(),
            cer_line: cer(&hyp_line, &s.gt),
            cer_seg: cer(&hyp_seg, &s.gt),
            wer_line: wer(&hyp_line, &s.gt),
            wer_seg: wer(&hyp_seg, &s.gt),
            hyp_line,
            hyp_seg,
            words_det,
            words_gt: s.words_gt,
        });
    }
    Ok(rows)
}

/// Agrégats d'un benchmark.
pub struct BenchSummary {
    pub n: usize,
    pub mean_cer_line: f64,
    pub mean_cer_seg: f64,
    pub mean_wer_line: f64,
    pub mean_wer_seg: f64,
    pub exact_line: usize,
    pub exact_seg: usize,
    pub mean_words_det: Option<f64>,
    pub mean_words_gt: Option<f64>,
}

pub fn summarize(rows: &[BenchRow]) -> BenchSummary {
    let n = rows.len();
    let m = |f: fn(&BenchRow) -> f64| if n == 0 { 0.0 } else { rows.iter().map(f).sum::<f64>() / n as f64 };
    let words_det: Vec<f64> = rows.iter().filter_map(|r| r.words_det.map(|w| w as f64)).collect();
    let words_gt: Vec<f64> = rows.iter().filter_map(|r| r.words_gt.map(|w| w as f64)).collect();
    BenchSummary {
        n,
        mean_cer_line: m(|r| r.cer_line),
        mean_cer_seg: m(|r| r.cer_seg),
        mean_wer_line: m(|r| r.wer_line),
        mean_wer_seg: m(|r| r.wer_seg),
        exact_line: rows.iter().filter(|r| r.cer_line == 0.0).count(),
        exact_seg: rows.iter().filter(|r| r.cer_seg == 0.0).count(),
        mean_words_det: if words_det.is_empty() {
            None
        } else {
            Some(words_det.iter().sum::<f64>() / words_det.len() as f64)
        },
        mean_words_gt: if words_gt.is_empty() {
            None
        } else {
            Some(words_gt.iter().sum::<f64>() / words_gt.len() as f64)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_connu() {
        assert_eq!(levenshtein("kitten".chars().collect::<Vec<_>>().as_slice(), "sitting".chars().collect::<Vec<_>>().as_slice()), 3);
        assert_eq!(cer("sitting", "kitten"), 3.0 / 6.0);
        assert_eq!(cer("", ""), 0.0);
        assert_eq!(cer("abc", ""), 1.0);
        assert_eq!(wer("a b c", "a x c"), 1.0 / 3.0);
        assert_eq!(wer("le chat", "le chat"), 0.0);
    }
}
