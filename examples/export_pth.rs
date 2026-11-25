//! Exporte tous les tenseurs d'un checkpoint torch en fichiers .f32 bruts +
//! meta.json (shapes), pour reconstruire le state_dict ailleurs (ex. PyTorch).
//! `cargo run --release --example export_pth -- <ckpt> <outdir>`

use std::env;

fn main() -> anyhow::Result<()> {
    let path = env::args().nth(1).expect("usage: export_pth <ckpt> <outdir>");
    let outdir = env::args().nth(2).expect("usage: export_pth <ckpt> <outdir>");
    std::fs::create_dir_all(&outdir)?;
    let mut archive = pthrs::PthArchive::open(&path)?;
    let tensors = archive.load_floating_f32()?;
    let mut meta = serde_json::Map::new();
    for (name, t) in &tensors {
        let (shape, values) = t.clone().into_shape_and_values()?;
        let fname = name.replace('.', "_") + ".f32";
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{outdir}/{fname}"), bytes)?;
        meta.insert(
            name.clone(),
            serde_json::json!({ "file": fname, "shape": shape }),
        );
    }
    std::fs::write(
        format!("{outdir}/meta.json"),
        serde_json::to_string_pretty(&serde_json::Value::Object(meta))?,
    )?;
    println!("exporté {} tenseurs dans {outdir}", tensors.len());
    Ok(())
}
