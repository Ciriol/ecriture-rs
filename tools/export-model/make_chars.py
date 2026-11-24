"""Génère chars.txt depuis le vocab.json du modèle Hugging Face.

chars.txt = un caractère par ligne, dans l'ordre des classes du modèle
(index 1..N). Le blank CTC (index 0) est réservé automatiquement par
`load_charset` côté Rust : on ne l'écrit pas ici.

Usage :
    uv run python make_chars.py [--repo Sachin-0001/crnn-iam-handwriting] [--out ../../chars.txt]
"""

import argparse
import json
import urllib.request
from pathlib import Path

DEFAULT_REPO = "Sachin-0001/crnn-iam-handwriting"
VOCAB_URL = "https://huggingface.co/{repo}/resolve/main/vocab.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "chars.txt",
        help="chemin de sortie (défaut: chars.txt à la racine du projet)",
    )
    args = p.parse_args()

    url = VOCAB_URL.format(repo=args.repo)
    print("téléchargement:", url)
    with urllib.request.urlopen(url) as r:
        vocab = json.load(r)

    c2i = vocab["char_to_idx"]
    blank_id = vocab.get("blank_id", 0)
    assert blank_id == 0 and all(i >= 1 for i in c2i.values()), (
        "blank CTC attendu à l'index 0, caractères à partir de 1"
    )
    chars = [c for c, _ in sorted(c2i.items(), key=lambda kv: kv[1])]
    args.out.write_text("\n".join(chars) + "\n")
    print(f"{len(chars)} caractères écrits dans {args.out}")


if __name__ == "__main__":
    main()
