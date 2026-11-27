//! LM de caractères (unigrammes + bigrammes avec backoff) pour la **shallow
//! fusion pendant le beam search CTC** — reprend l'idée de
//! [githubharald/CTCDecoder](https://github.com/githubharald/CTCDecoder)
//! (`beam_search` + `LanguageModel` : bigrammes de caractères appliqués à
//! chaque extension de préfixe, score normalisé par la longueur en fin de
//! beam), réimplémenté en pur Rust.
//!
//! Le LM est construit depuis le lexique de langue embarqué
//! (`assets/lexique-fr.txt`, trié par fréquence décroissante) : chaque mot
//! est pondéré par `1 / rang` (loi de Zipf), les frontières de mots sont
//! modélisées par un espace si `' '` fait partie du charset du modèle.
//! Seuls les caractères du charset du modèle sont pris en compte.

use std::collections::HashMap;

/// LM de caractères sur le charset d'un modèle : `uni[c]` = ln P(c),
/// `bi[c][d]` = ln P(d | c) avec backoff vers `uni[d]` quand le bigramme
/// n'a jamais été vu.
#[derive(Clone)]
pub struct CharLm {
    n: usize,
    index: HashMap<char, usize>,
    uni: Vec<f64>,
    bi: Vec<f64>,
    /// ln P pour un caractère hors charset (plancher).
    floor: f64,
}

impl CharLm {
    /// Construit le LM depuis une liste de mots ordonnés par fréquence
    /// décroissante, restreint au `charset` du modèle.
    pub fn from_words(words: &[String], charset: &[char]) -> Self {
        let n = charset.len();
        let mut index = HashMap::with_capacity(n);
        for (i, &c) in charset.iter().enumerate() {
            index.insert(c, i);
        }
        let mut uni_c = vec![0f64; n];
        let mut bi_c = vec![0f64; n * n];
        let mut total = 0f64;
        for (rank, w) in words.iter().enumerate() {
            // Pondération Zipf sur le rang (le lexique est trié par fréquence).
            let wt = 1.0 / (rank as f64 + 1.0);
            // On parcourt le mot suivi d'un séparateur ' ' (si dans le
            // charset) pour modéliser les frontières de mots.
            let mut prev: Option<usize> = None;
            for c in w.chars().chain(std::iter::once(' ')) {
                let Some(&ci) = index.get(&c) else { continue };
                uni_c[ci] += wt;
                total += wt;
                if let Some(p) = prev {
                    bi_c[p * n + ci] += wt;
                }
                prev = Some(ci);
            }
        }
        // Lissage add-one : P(c) = (count(c) + 1) / (total + n).
        let uni: Vec<f64> = uni_c
            .iter()
            .map(|&c| ((c + 1.0) / (total + n as f64)).ln())
            .collect();
        let floor = (1.0 / (total + n as f64)).ln();
        // Bigrammes conditionnels avec backoff vers l'unigramme.
        let mut bi = vec![0f64; n * n];
        for c in 0..n {
            let row_total: f64 = bi_c[c * n..(c + 1) * n].iter().sum();
            for d in 0..n {
                bi[c * n + d] = if row_total > 0.0 && bi_c[c * n + d] > 0.0 {
                    (bi_c[c * n + d] / row_total).ln()
                } else {
                    uni[d]
                };
            }
        }
        Self {
            n,
            index,
            uni,
            bi,
            floor,
        }
    }

    /// LM depuis un lexique (mots ordonnés par fréquence).
    pub fn from_lexicon(lex: &crate::lexicon::Lexicon, charset: &[char]) -> Self {
        Self::from_words(lex.words_ordered(), charset)
    }

    /// LM français embarqué (`assets/lexique-fr.txt`) sur le charset donné.
    pub fn embedded_fr(charset: &[char]) -> Self {
        let words: Vec<String> = include_str!("../assets/lexique-fr.txt")
            .lines()
            .map(|l| l.trim().to_lowercase().replace('’', "'"))
            .filter(|l| !l.is_empty())
            .collect();
        Self::from_words(&words, charset)
    }

    /// ln P(c | prev) : bigramme si `prev` est connu, sinon unigramme ;
    /// plancher pour un caractère hors charset.
    pub fn logp(&self, prev: Option<char>, c: char) -> f64 {
        let Some(&ci) = self.index.get(&c) else {
            return self.floor;
        };
        match prev.and_then(|p| self.index.get(&p).copied()) {
            Some(pi) => self.bi[pi * self.n + ci],
            None => self.uni[ci],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bigram_vu_batte_backoff() {
        // charset: 0=blank('\0'), 1='a', 2='b', 3='c'
        let charset: Vec<char> = vec!['\u{0}', 'a', 'b', 'c'];
        let lm = CharLm::from_words(&["ab".to_string()], &charset);
        // "ab" vu -> P(b|a) = 1 -> ln = 0, bien au-dessus du backoff pour 'c'
        // (backoff = unigramme de 'c', lissé add-one -> ln(1/6) ~ -1.79).
        assert!((lm.logp(Some('a'), 'b') - 0.0).abs() < 1e-9);
        assert!(lm.logp(Some('a'), 'c') < lm.logp(Some('a'), 'b') - 1.0);
        // unigramme
        assert!(lm.logp(None, 'a') > lm.logp(None, 'c'));
        // caractère hors charset -> plancher (ln(1/(total+n)) ici ~ -1.79)
        assert_eq!(lm.logp(None, '€'), lm.floor);
        assert!((lm.floor - (1.0f64 / 6.0).ln()).abs() < 1e-9);
    }

    #[test]
    fn espace_modelise_les_frontieres() {
        let charset: Vec<char> = vec!['\u{0}', 'a', 'b', ' '];
        let lm = CharLm::from_words(&["ab".to_string(), "ba".to_string()], &charset);
        // après un mot, l'espace est très probable
        assert!(lm.logp(Some('b'), ' ') > lm.logp(Some('b'), 'a'));
    }

    #[test]
    fn lm_embarque_francais_est_coherent() {
        let charset: Vec<char> = ('a'..='z').chain([' ', '\u{0}']).collect();
        let lm = CharLm::embedded_fr(&charset);
        // "le" est très fréquent en français : P(e|l) >> P(z|l)
        assert!(lm.logp(Some('l'), 'e') > lm.logp(Some('l'), 'z'));
        // unigramme : 'e' plus fréquent que 'z'
        assert!(lm.logp(None, 'e') > lm.logp(None, 'z'));
        // 'q' est quasi toujours suivi de 'u' en français
        assert!(lm.logp(Some('q'), 'u') > lm.logp(Some('q'), 'a'));
    }

    #[test]
    fn depuis_lexique_garde_l_ordre() {
        let lex = crate::lexicon::Lexicon::from_text("un\ndeux\ntrois\n");
        let charset: Vec<char> = vec!['\u{0}', 'a', 'b', 'c', 'd', 'e', 'i', 'o', 'r', 's', 't', 'u', 'x', 'n'];
        let lm = CharLm::from_lexicon(&lex, &charset);
        // pondération Zipf sur le rang : 'u' (dans "un" rang 1 + "deux" rang 2)
        // plus fréquent que 'x' (seulement dans "deux", rang 2)
        assert!(lm.logp(None, 'u') > lm.logp(None, 'x'));
    }
}
