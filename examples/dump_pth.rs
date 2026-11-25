//! Liste les tenseurs (nom + shape) d'un checkpoint torch (.pth / .bin).
//! `cargo run --release --example dump_pth -- ~/modeles/crnn-fr/pytorch_model.bin`

use std::env;

fn main() -> anyhow::Result<()> {
    let path = env::args().nth(1).expect("usage: dump_pth <fichier.pth>");
    let mut archive = pthrs::PthArchive::open(&path)?;
    let tensors = archive.load_floating_f32()?;
    let mut names: Vec<_> = tensors.keys().cloned().collect();
    names.sort();
    for name in names {
        let t = &tensors[&name];
        let (shape, values) = t.clone().into_shape_and_values()?;
        println!("{name} {:?} ({} valeurs)", shape, values.len());
    }
    Ok(())
}
