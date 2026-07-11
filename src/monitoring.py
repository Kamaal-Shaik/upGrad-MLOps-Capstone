"""monitoring.py — Stage 4: statistical + embedding drift + confidence monitoring.

Implement the three drift signals against the clean reference baseline:
  1. statistical drift  — Evidently DataDriftPreset + PSI on image features
  2. embedding drift    — PSI on ResNet-embedding distance-to-centroid distribution
  3. confidence         — mean predicted confidence reference vs current
Use a corrupted copy of clean images as the simulated "current" production batch.
Outputs drift_report.html + drift_summary.json.   Run: python -m src.monitoring

Embedding drift (TODO 4) — see the conceptual walkthrough in
Operations_Monitoring_and_Evidence.ipynb (Stage 4.3):
  1. feature extraction  — penultimate 512-dim ResNet embedding (model.EmbeddingExtractor)
  2. embedding generation — embeddings for reference + current batches
  3. feature-space compare — reduce each to distance-to-reference-centroid (one distribution per batch)
  4. drift calculation   — PSI between the two distance distributions (> ~0.10 => drifted)
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep
from src.model import load_model, EmbeddingExtractor


def psi(reference, current, bins: int = 10) -> float:
    # TODO 4: Population Stability Index between two 1-D distributions (quantile bins).
    # raise NotImplementedError

    ref = np.array(reference)
    cur = np.array(current)

    # Quantile bins from reference
    quantiles = np.percentile(ref, np.linspace(0, 100, bins+1))
    ref_counts, _ = np.histogram(ref, bins=quantiles)
    cur_counts, _ = np.histogram(cur, bins=quantiles)

    ref_dist = ref_counts / len(ref)
    cur_dist = cur_counts / len(cur)

    psi_val = 0.0
    for r, c in zip(ref_dist, cur_dist):
        if r > 0 and c > 0:
            psi_val += (r - c) * np.log(r / c)

    return psi_val


def corrupt(img: Image.Image) -> Image.Image:
    # TODO 4: simulate camera/lighting drift (brightness/blur/rotate/noise via config.DRIFT_SIM).
    # raise NotImplementedError

    drift = config.DRIFT_SIM
    out = img.copy()

    if drift.get("brightness", 0) != 0:
        out = ImageEnhance.Brightness(out).enhance(1 + drift["brightness"])

    if drift.get("blue", 0) != 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=drift["blur"]))

    if drift.get("rotate", 0) != 0:
        out = out.rotate(drift["rotate"])

    if drift.get("noise", 0) != 0:
        arr = np.array(out)
        noise = np.random.normal(0, drift["noise"], arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)

    return out



def run() -> dict:
    # TODO 4: build reference (clean) + current (corrupted) batches → features, embeddings,
    #         mean confidence. Run Evidently DataDriftPreset + PSI; embedding PSI on
    #         distance-to-centroid; confidence drop. Write drift_summary.json + drift_report.html
    #         and set retrain_recommended from the configured thresholds.
    # raise NotImplementedError

    # root = data_prep.find_data_root()
    ref_items = [(Path(d["path"]), d["label"]) for d in json.load(open(config.SPLIT_DIR / "v1" / "val.json"))["files"]]

    # Reference batch (clean)
    ref_imgs = [Image.open(p).convert("L") for p, _ in ref_items]
    ref_feats = [data_prep.image_features(img) for img in ref_imgs]

    # Current batch (corrupted)
    cur_imgs = [corrupt(img) for img in ref_imgs]
    cur_feats = [data_prep.image_features(img) for img in cur_imgs]

    # Statistical drift - Evidently
    ref_df = pd.DataFrame(ref_feats)
    cur_df = pd.DataFrame(cur_feats)

    from evidently.future.report import Report
    from evidently.future.presets import DataDriftPreset
    report = Report(metrics=[DataDriftPreset()])
    eval_res = report.run(reference_data=ref_df, current_data=cur_df)
    eval_res.save_html(str(config.ARTIFACT_DIR / "drift_report.html"))

    # PSI per feature
    psi_scores = {f: psi(ref_df[f], cur_df[f]) for f in ref_df.columns}

    # Embedding drift
    net = load_model(config.MODEL_PATH).to(config.DEVICE)
    extractor = EmbeddingExtractor(net)
    ref_emb = np.vstack([extractor(data_prep.get_transforms(False)(img).unsqueeze(0).to(config.DEVICE)).cpu().numpy() for img in ref_imgs])
    cur_emb = np.vstack([extractor(data_prep.get_transforms(False)(img).unsqueeze(0).to(config.DEVICE)).cpu().numpy() for img in cur_imgs])

    centroid = ref_emb.mean(axis=0)
    ref_dist = np.linalg.norm(ref_emb - centroid, axis=1)
    cur_dist = np.linalg.norm(cur_emb - centroid, axis=1)
    emb_psi = psi(ref_dist, cur_dist)

    # Confidence drift
    import torch.nn.functional as F
    net.eval()
    ref_conf = []
    cur_conf = []
    for batch in [ref_imgs, cur_imgs]:
        for img in batch:
            xb = data_prep.get_transforms(False)(img).unsqueeze(0).to(config.DEVICE)
            probs = F.softmax(net(xb), dim=1)
            conf = probs.max().item()
            if batch is ref_imgs:
                ref_conf.append(conf)
            else:
                cur_conf.append(conf)
    mean_ref_conf = float(np.mean(ref_conf))
    mean_cur_conf = float(np.mean(cur_conf))
    conf_drop = mean_ref_conf - mean_cur_conf

    # Decision
    retrain_recommended = (
        any(v > config.PSI_THRESHOLD for v in psi_scores.values()) or
        emb_psi > config.EMBEDDING_DRIFT_THRESHOLD or
        conf_drop > config.CONFIDENCE_DROP_THRESHOLD
    )

    summary = {
        "psi_scores": psi_scores, "embedding_psi": emb_psi, "confidence_ref": mean_ref_conf,
        "confidence_cur": mean_cur_conf, "confidence_drop": conf_drop, "retrain_recommended":retrain_recommended
    }

    (config.ARTIFACT_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    run()
