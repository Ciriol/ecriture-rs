//! Bibliothèque ecriture-rs : capture d'encre, modèles de reconnaissance
//! d'écriture manuscrite (CRNN-CTC, classifieur de chiffres MNIST) et outils
//! associés. Le binaire `ecriture-rs` (src/main.rs) et les outils
//! (src/bin/*.rs) s'appuient sur cette lib.

pub mod charlm;
pub mod digits;
pub mod doctr;
pub mod ink;
pub mod lexicon;
pub mod model;
pub mod pylaia;
pub mod recognizer;
pub mod tablet;
