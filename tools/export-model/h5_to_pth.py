#!/usr/bin/env python3
"""Convertit le checkpoint Keras `weights.h5` de chunkyjasper/IAMhwr (ONNET :
TDNN + 4x BiLSTM + CTC, IAM-OnDB online, CER ~6.2 %) en un `.pth` PyTorch
standard, chargeable en pur Rust via pthrs + candle (backend `online.rs`).

Usage (venv de tools/export-model, h5py + torch installes) :
  .venv/bin/python tools/export-model/h5_to_pth.py <weights.h5> <out.pth> <out-config.json> [--ref-xml sample.xml --ref-out /tmp/online_ref]

Produit aussi (avec --ref-xml) une entree/sortie de reference pour le test de
parite Rust vs torch : <ref-out>_input.f32 ([T,6]), <ref-out>_logits.f32
([T',83] log_softmax), <ref-out>_meta.json.

Schéma des noms dans le .pth (identique a un nn.Module PyTorch standard) :
  convs.{i}.weight/bias           Conv1d (out,in,k), padding = k//2 ("same", k impair)
  bns.{i}.weight/bias/running_mean/running_var   BatchNorm1d, eps = 1e-3 (Keras)
  lstm.weight_ih_l{k}[_reverse] ...              LSTM bidirectionnel 4 couches, hidden 60
  dense.weight/bias               Linear (83, 120)

Conversion des poids :
  Conv1D Keras kernel (k, in, out) -> torch (out, in, k) = transpose(2,1,0)
  LSTM Keras/CuDNN kernel (in, 4H), recurrent_kernel (H, 4H), bias (2*4H,) =
  [bias_ih, bias_hh] ; ordre des gates identique (i, f, c, o) -> transpose seul.
  Dense kernel (in, out) -> torch (out, in) = T.
"""

import json
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch
import torch.nn as nn

# Charset de IAMhwr (hwr/constants.py DATA.CHARS, 83 elements dont le dernier
# ('%') est ecrase par le blank CTC) ; les deux chaines vides (ga/do) deviennent
# des placeholders \x01/\x02. Classes 0..81 = caracteres, 82 = blank.
CHARS = (
    ['!', '"', '\x01', '\x02', '&', "'", '/', '(', ')', '[', ']', '*', ',', '-',
     '+', '.', ' ', ':', ';', '?']
    + [str(d) for d in range(10)]
    + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    + [chr(c) for c in range(ord('a'), ord('z') + 1)]
    + ['%']
)
assert len(CHARS) == 83
NUM_CLASSES = 83
BLANK = 82
CHARSET = "".join(CHARS[:BLANK])  # 82 caracteres affichables

CONVS = [(6, 60, 7), (60, 90, 5), (90, 120, 5), (120, 120, 3), (120, 160, 3), (160, 200, 3)]
POOL_AFTER = [2, 5]  # AveragePooling1D(2) apres conv2 et conv5
LSTM_LAYERS = 4
LSTM_HIDDEN = 60
BN_EPS = 1e-3  # Keras BatchNormalization par defaut (torch: 1e-5)

# Preprocessing SCHEME6 de IAMhwr (hwr/constants.py)
SCHEME6 = {"slope_correction": True, "normalize": True, "down_cos": 0.975,
           "resample_distance": 0.37, "up_sample": 9}


def convert(h5_path, out_pth, out_cfg):
    f = h5py.File(h5_path, 'r')

    def g(path):
        return np.asarray(f[path][...], dtype=np.float32)

    sd = {}
    for i, (cin, cout, k) in enumerate(CONVS):
        name = "conv1d" if i == 0 else f"conv1d_{i}"
        kernel = g(f"{name}/{name}/kernel:0")  # (k, in, out)
        bias = g(f"{name}/{name}/bias:0")      # (out,)
        sd[f"convs.{i}.weight"] = torch.from_numpy(kernel.transpose(2, 1, 0).copy())
        sd[f"convs.{i}.bias"] = torch.from_numpy(bias.copy())
    for i in range(7):
        name = "batch_normalization" if i == 0 else f"batch_normalization_{i}"
        sd[f"bns.{i}.weight"] = torch.from_numpy(g(f"{name}/{name}/gamma:0").copy())
        sd[f"bns.{i}.bias"] = torch.from_numpy(g(f"{name}/{name}/beta:0").copy())
        sd[f"bns.{i}.running_mean"] = torch.from_numpy(g(f"{name}/{name}/moving_mean:0").copy())
        sd[f"bns.{i}.running_var"] = torch.from_numpy(g(f"{name}/{name}/moving_variance:0").copy())
    # 8 cu_dnnlstm = 4 couches x (fwd, bwd) ; couche k -> l{k} (fwd) / l{k}_reverse (bwd)
    for k in range(LSTM_LAYERS):
        fwd = "cu_dnnlstm" if 2 * k == 0 else f"cu_dnnlstm_{2 * k}"
        bwd = f"cu_dnnlstm_{2 * k + 1}"
        for tag, src in ((f"l{k}", fwd), (f"l{k}_reverse", bwd)):
            kernel = g(f"{src}/{src}/kernel:0")            # (in, 4H)
            recurrent = g(f"{src}/{src}/recurrent_kernel:0")  # (H, 4H)
            bias = g(f"{src}/{src}/bias:0")                # (2*4H,) = [b_ih, b_hh]
            h = recurrent.shape[0]
            sd[f"lstm.weight_ih_{tag}"] = torch.from_numpy(kernel.T.copy())     # (4H, in)
            sd[f"lstm.weight_hh_{tag}"] = torch.from_numpy(recurrent.T.copy())  # (4H, H)
            sd[f"lstm.bias_ih_{tag}"] = torch.from_numpy(bias[:4 * h].copy())
            sd[f"lstm.bias_hh_{tag}"] = torch.from_numpy(bias[4 * h:].copy())
    sd["dense.weight"] = torch.from_numpy(g("dense/dense/kernel:0").T.copy())  # (83, 120)
    sd["dense.bias"] = torch.from_numpy(g("dense/dense/bias:0").copy())

    torch.save(sd, out_pth)
    cfg = {
        "arch": "onnet-iamon",
        "source": "https://github.com/chunkyjasper/IAMhwr (weights.h5, CER ~6.2% IAM-OnDB)",
        "input_features": 6,
        "feature_names": ["x", "y", "dx", "dy", "pen_down", "pen_up"],
        "convs": [[o, k] for _, o, k in CONVS],
        "pool_after": POOL_AFTER,
        "lstm_layers": LSTM_LAYERS,
        "lstm_hidden": LSTM_HIDDEN,
        "bn_eps": BN_EPS,
        "num_classes": NUM_CLASSES,
        "blank": BLANK,
        "charset": CHARSET,
        "preprocess": SCHEME6,
    }
    with open(out_cfg, "w") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)
    print(f"poids ecrits: {out_pth} ({len(sd)} tenseurs), config: {out_cfg}")
    return sd


class ONNET(nn.Module):
    """Modele torch de reference, meme forward que le Keras ONNET."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(i, o, k, padding=k // 2) for i, o, k in CONVS])
        self.bns = nn.ModuleList([nn.BatchNorm1d(o, eps=BN_EPS) for _, o, _ in CONVS]
                                 + [nn.BatchNorm1d(2 * LSTM_HIDDEN, eps=BN_EPS)])
        self.lstm = nn.LSTM(CONVS[-1][1], LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                            bidirectional=True, batch_first=True)
        self.dense = nn.Linear(2 * LSTM_HIDDEN, NUM_CLASSES)

    def forward(self, x):  # x: [B, T, 6]
        x = x.transpose(1, 2)  # [B, 6, T]
        for i, conv in enumerate(self.convs):
            x = torch.relu(self.bns[i](conv(x)))
            if i in POOL_AFTER:
                x = torch.avg_pool1d(x, 2)
        x = x.transpose(1, 2)  # [B, T', C]
        x, _ = self.lstm(x)    # [B, T', 2H]
        x = self.bns[6](x.transpose(1, 2)).transpose(1, 2)
        return torch.log_softmax(self.dense(x), dim=-1)


# ---------- preprocessing SCHEME6 (recopie fidele de hwr/data/datarep.py) ----------

def read_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    points = []  # [stroke, x, y]
    for sid, stroke in enumerate(root.iter("Stroke")):
        for p in stroke.iter("Point"):
            points.append([sid, float(p.get("x")), float(p.get("y"))])
    return points


def stroke_groups(points):
    groups = {}
    for p in points:
        groups.setdefault(p[0], []).append(p)
    return list(groups.values())  # ordre d'apparition


def line(p1, p2):
    dx, dy = p2[1] - p1[1], p2[2] - p1[2]
    return dx, dy, float(np.hypot(dx, dy))


def slope_correction(points):
    x = np.array([p[1] for p in points])
    y = np.array([p[2] for p in points])
    f = np.poly1d(np.polyfit(x, y, 1))
    rad = np.arctan2(f(x.max()) - f(x.min()), x.max() - x.min())
    c, s = np.cos(rad), np.sin(rad)
    for p in points:
        px, py = p[1], p[2]
        p[1], p[2] = px * c + py * s, -px * s + py * c


def all_lines(points):
    out = []
    for g in stroke_groups(points):
        for i in range(len(g) - 1):
            dx, dy, l = line(g[i], g[i + 1])
            out.append((g[i], g[i + 1], l))
    return out


def normalize(points):
    lines = all_lines(points)
    sum_l = sum(l[2] for l in lines) or 1.0
    mean_x = sum(l[2] * (l[0][1] + l[1][1]) / 2 for l in lines) / sum_l
    mean_y = sum(l[2] * (l[0][2] + l[1][2]) / 2 for l in lines) / sum_l
    var_y = sum(l[2] / 3 * ((l[1][2] - mean_y) ** 2 + (l[0][2] - mean_y) ** 2
                            + (l[0][2] - mean_y) * (l[1][2] - mean_y)) for l in lines)
    sd_y = np.sqrt(var_y / sum_l) or 1.0
    for p in points:
        p[1] = (p[1] - mean_x) / sd_y
        p[2] = (p[2] - mean_y) / sd_y


def down_sample_angle(points, cos_th):
    ret = []
    for s in stroke_groups(points):
        removed = 0
        for i in range(len(s)):
            if i == 0 or i == len(s) - 1:
                removed = 0
                ret.append(s[i])
            else:
                dx1, dy1, l1 = line(s[i - 1 - removed], s[i])
                dx2, dy2, l2 = line(s[i], s[i + 1])
                cs = np.inf if l1 * l2 < 1e-5 else (dx1 * dx2 + dy1 * dy2) / (l1 * l2)
                if cs < cos_th:
                    removed = 0
                    ret.append(s[i])
                else:
                    removed += 1
    return ret


def resample_distance(points, d):
    ret = []
    for s in stroke_groups(points):
        if not s:
            continue
        ret.append(s[0])
        for i in range(1, len(s)):
            dx, dy, l = line(ret[-1], s[i])
            if l > d:
                f = d / l
                for j in range(int(l / d)):
                    pc = f * (j + 1)
                    ret.append([ret[-1][0], ret[-1][1] + dx * pc, ret[-1][2] + dy * pc])
            elif l == d:
                ret.append(s[i])
            elif i < d:
                continue
        ret.append(s[-1])
    return ret


def up_sample_short_stroke(points, n):
    out = []
    for s in stroke_groups(points):
        if len(s) < n:
            if len(s) == 1:
                out.extend([s[0] for _ in range(n)])
            else:
                total = sum(line(s[i], s[i + 1])[2] for i in range(len(s) - 1))
                out.extend(resample_distance(s, total / n))
        else:
            out.extend(s)
    return out


def preprocess(points, scheme=SCHEME6):
    pts = [list(p) for p in points]
    if scheme["slope_correction"]:
        slope_correction(pts)
    if scheme["normalize"]:
        normalize(pts)
    if scheme.get("down_cos", 1) < 1:
        pts = down_sample_angle(pts, scheme["down_cos"])
    if scheme.get("resample_distance", 0) > 0:
        pts = resample_distance(pts, scheme["resample_distance"])
    if scheme.get("up_sample", 0) > 0:
        pts = up_sample_short_stroke(pts, scheme["up_sample"])
    return pts


def features(points):
    """[x, y, dx, dy, pen_down, pen_up] par segment, dernier segment de chaque
    stroke avec pen_up=1 (Line.get_features de datarep.py)."""
    feats = []
    for g in stroke_groups(points):
        for i in range(len(g) - 1):
            eos = i == len(g) - 2
            dx, dy, _ = line(g[i], g[i + 1])
            feats.append([g[i][1], g[i][2], dx, dy, 0.0 if eos else 1.0, 1.0 if eos else 0.0])
    return np.array(feats, dtype=np.float32)


def ctc_greedy(log_softmax):  # [T, C] -> texte
    idx = log_softmax.argmax(-1)
    out, prev = [], BLANK
    for i in idx:
        i = int(i)
        if i != BLANK and i != prev and i < BLANK:
            out.append(CHARS[i])
        prev = i
    return "".join(out)


def main():
    h5 = sys.argv[1]
    out_pth = sys.argv[2]
    out_cfg = sys.argv[3]
    sd = convert(h5, out_pth, out_cfg)

    ref_xml = None
    ref_out = "/tmp/online_ref"
    args = sys.argv[4:]
    if "--ref-xml" in args:
        ref_xml = args[args.index("--ref-xml") + 1]
    if "--ref-out" in args:
        ref_out = args[args.index("--ref-out") + 1]

    if ref_xml:
        model = ONNET()
        model.load_state_dict(sd)
        model.eval()
        pts = preprocess(read_xml(ref_xml))
        x = torch.from_numpy(features(pts))[None]  # [1, T, 6]
        with torch.no_grad():
            logits = model(x)[0]  # [T', 83] log_softmax
        pred = ctc_greedy(logits.numpy())
        print(f"{ref_xml}: T={x.shape[1]} -> T'={logits.shape[0]}, prediction greedy: {pred!r}")
        np.asarray(x[0].numpy(), dtype=np.float32).tofile(ref_out + "_input.f32")
        np.asarray(logits.numpy(), dtype=np.float32).tofile(ref_out + "_logits.f32")
        with open(ref_out + "_meta.json", "w") as fh:
            json.dump({"T": int(x.shape[1]), "Tprime": int(logits.shape[0]), "C": NUM_CLASSES,
                       "prediction": pred}, fh)


if __name__ == "__main__":
    main()
