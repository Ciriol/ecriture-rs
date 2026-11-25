//! Petit lexique de langue (français par défaut, embarqué dans le binaire)
//! utilisé pour rescorer les N-best hypothèses du beam CTC (shallow fusion) :
//! une hypothèse dont les mots appartiennent au lexique est préférée à une
//! hypothèse de même vraisemblance CTC mais hors lexique.

use std::collections::HashSet;
use std::path::Path;

/// Lexique = ensemble de mots (minuscules), issu d'une liste de fréquences.
#[derive(Clone)]
pub struct Lexicon {
    words: HashSet<String>,
}

impl Lexicon {
    /// Construit un lexique depuis un texte "un mot par ligne".
    pub fn from_text(text: &str) -> Self {
        let words = text
            .lines()
            .map(|l| l.trim().to_lowercase().replace('’', "'"))
            .filter(|l| !l.is_empty())
            .collect();
        Self { words }
    }

    /// Charge un lexique depuis un fichier texte (un mot par ligne).
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(Self::from_text(&text))
    }

    /// Lexique français embarqué (top ~49k mots d'un corpus de fréquences,
    /// formes fléchies incluses) : `assets/lexique-fr.txt`.
    pub fn embedded_fr() -> Self {
        Self::from_text(include_str!("../assets/lexique-fr.txt"))
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    pub fn contains(&self, word: &str) -> bool {
        self.words.contains(&word.to_lowercase().replace('’', "'"))
    }

    /// Extrait le "cœur" d'un token : lettres uniquement (apostrophes et
    /// tirets internes conservés, ponctuation externe ignorée).
    /// `"l'"` -> "l", `"(porte-clés),"` -> "porte-clés", `"39"` -> "".
    fn core(token: &str) -> String {
        token
            .trim()
            .chars()
            .skip_while(|c| !c.is_alphabetic())
            .take_while(|c| c.is_alphabetic() || *c == '\'' || *c == '-' || *c == '’')
            .collect::<String>()
            .to_lowercase()
            .replace('’', "'")
    }

    /// Proportion de caractères alphabétiques du texte qui appartiennent à un
    /// mot du lexique (0.0..1.0). Les tokens sans lettres (nombres,
    /// ponctuation seule) sont neutres : exclus du total.
    pub fn coverage(&self, text: &str) -> f64 {
        let mut total = 0usize;
        let mut known = 0usize;
        for tok in text.split_whitespace() {
            let c = Self::core(tok);
            if c.is_empty() {
                continue;
            }
            let n = c.chars().count();
            total += n;
            if self.words.contains(&c) {
                known += n;
            }
        }
        if total == 0 {
            0.0
        } else {
            known as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex() -> Lexicon {
        Lexicon::from_text("le\nchat\nchien\naujourd'hui\nmange\nporte-clés\nêtre\n")
    }

    #[test]
    fn coverage_mots_connus_et_inconnus() {
        let l = lex();
        assert!((l.coverage("le chat mange") - 1.0).abs() < 1e-9);
        // "le"(2) + "mange"(5) connus, "zqxwv"(5) inconnu -> 7/12
        assert!((l.coverage("le zqxwv mange") - 7.0 / 12.0).abs() < 1e-9);
        assert_eq!(l.coverage("zqxwv"), 0.0);
        assert_eq!(l.coverage(""), 0.0);
    }

    #[test]
    fn coverage_ignore_ponctuation_et_nombres() {
        let l = lex();
        // ponctuation externe stripped, nombres neutres
        assert!((l.coverage("le chat, mange. 39 !") - 1.0).abs() < 1e-9);
        assert!((l.coverage("aujourd'hui porte-clés") - 1.0).abs() < 1e-9);
        assert!((l.coverage("(être)") - 1.0).abs() < 1e-9);
        // casse ignorée
        assert!((l.coverage("LE Chat") - 1.0).abs() < 1e-9);
        // apostrophe courbe normalisée
        assert!((l.coverage("aujourd’hui") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn lexique_embarque_charge() {
        let l = Lexicon::embedded_fr();
        assert!(l.len() > 40_000);
        assert!(l.contains("français"));
        assert!(l.contains("aujourd'hui"));
        assert!(!l.contains("zqxwvkj"));
    }
}
