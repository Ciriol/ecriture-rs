"""Exporte le CRNN IAM (Hugging Face) en model.onnx pour ecriture-rs.

Entrée du graphe :  [1, 1, 32, W] f32, pixels normalisés [-1, 1] (W dynamique)
Sortie du graphe :  [1, T, num_classes] log-probabilités, décodage CTC glouton
                    côté Rust (recognizer.rs), blank = index 0.

Usage :
    uv run python export_onnx.py [--repo Sachin-0001/crnn-iam-handwriting] [--out ../../model.onnx]
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import torch

DEFAULT_REPO = "Sachin-0001/crnn-iam-handwriting"
REPO_FILE_URL = "https://huggingface.co/{repo}/resolve/main/{name}"


def hf_download(repo: str, name: str, dest_dir: Path) -> Path:
    dest = dest_dir / name
    if not dest.exists():
        url = REPO_FILE_URL.format(repo=repo, name=name)
        print("téléchargement:", url)
        dest_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--workdir", type=Path, default=Path(__file__).parent / ".hf")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "model.onnx",
        help="chemin de sortie (défaut: model.onnx à la racine du projet)",
    )
    args = p.parse_args()

    hf_download(args.repo, "model.py", args.workdir)
    hf_download(args.repo, "model.pth", args.workdir)
    config = json.load(open(hf_download(args.repo, "config.json", args.workdir)))

    sys.path.insert(0, str(args.workdir))
    from model import CRNN

    model = CRNN(num_classes=config["num_classes"], hidden_size=config["hidden_size"])
    state = torch.load(args.workdir / "model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    class Wrapper(torch.nn.Module):
        """Le CRNN sort [T, N, C] ; recognizer.rs attend des logits [1, T, C]."""

        def __init__(self, m: torch.nn.Module):
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.m(x).permute(1, 0, 2)

    wrapper = Wrapper(model).eval()
    dummy = torch.zeros(1, 1, config["image_height"], 128)
    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {3: "W"}, "logits": {1: "T"}},
        opset_version=17,
    )
    print(f"modèle exporté: {args.out}")

    import onnx

    m = onnx.load(str(args.out))
    onnx.checker.check_model(m)
    print(
        "graphe ONNX valide ; entrée:",
        [i.name for i in m.graph.input],
        "sortie:",
        [o.name for o in m.graph.output],
    )


if __name__ == "__main__":
    main()
