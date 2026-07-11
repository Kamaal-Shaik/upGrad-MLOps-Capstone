"""retrain.py — Stage 4: drift-triggered retraining, version compare, rollback.

Implement: read drift_summary.json; if retraining is recommended, measure the production
model on a drifted batch, train a drift-augmented candidate, compare, then PROMOTE the
candidate to @production only if it improves (within PROMOTE_EPSILON) else ROLL BACK.
Record retraining_decision.json + manage MLflow registry versions.   Run: python -m src.retrain
"""
from __future__ import annotations

import json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep, evaluate
from src.model import build_model, trainable_parameters, load_model, save_model
from src.monitoring import corrupt
from src.train import _subsample, set_seed, class_weights


def main() -> int:
    drift = json.loads((config.ARTIFACT_DIR / "drift_summary.json").read_text())
    if not drift.get("retrain_recommended"):
        # TODO 4: record a 'no_retrain' decision and return.
        # raise NotImplementedError
        decision = {
            "action": "no_retrain", "reason": "Drift thresholds not exceeded", "drift_summary": drift
        }

        (config.ARTIFACT_DIR / "retraining_decision.json").write_text(json.dumps(decision, indent=2))
        return 0

    # TODO 4: evaluate current production on a fully-drifted eval batch (corrupt_frac=1.0).
    # TODO 4: train a drift-augmented candidate (corrupt_frac~0.4, few epochs, capped).
    # TODO 4: compare candidate vs production F1 on the drifted batch.
    # TODO 4: promote candidate→@production iff cand_f1 >= prod_f1 + PROMOTE_EPSILON, else
    #         rollback (keep incumbent). Register the candidate version; move/keep the alias.
    # TODO 4: write retraining_decision.json (scores, action, version history).
    # raise NotImplementedError("Implement retraining + version compare + rollback")

    # Evaluate production on fully drifted batch
    prod_model = load_model(config.MODEL_PATH).to(config.DEVICE)
    prod_model.eval()
    test_items = [(Path(d["path"]), d["label"]) for d in json.load(open(config.SPLIT_DIR / "v1" / "test.json"))["files"]]
    # drifted_ds = [(data_prep.get_transforms(False)(corrupt(Image.open(p).convert("L"))), cls) for p, cls in test_items]
    drifted_ds = []
    for p, cls in test_items:
        img = Image.open(p).convert("L")
        if random.random() < config.DRIFT_SHARE_THRESHOLD: # can be config.DRIFT_SHARE_THRESHOLD
            img = corrupt(img)
        drifted_ds.append((data_prep.get_transforms(False)(img), cls))
    drift_loader = DataLoader(drifted_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    y_true, y_pred, y_prob = evaluate.predict(prod_model, drift_loader)
    prod_metrics = evaluate.compute_metrics(y_true, y_pred, y_prob)
    prod_f1 = prod_metrics["f1_defect"]

    # Train candidate on drift-augmented data (corrupt_frac ~0.4)
    train_items = [(Path(d["path"]), d["label"]) for d in json.load(open(config.SPLIT_DIR / "v1" / "train.json"))["files"]]
    if config.MAX_TRAIN_IMAGES > 0 and len(train_items) > config.MAX_TRAIN_IMAGES:
        train_items = train_items[:config.MAX_TRAIN_IMAGES]

    aug_ds = []
    for p, cls in train_items:
        if not p.exists():
            print("Missing file:", {p})
            continue
        img = Image.open(p).convert("L")
        if random.random() < config.DRIFT_SHARE_THRESHOLD:
            img = corrupt(img)
        aug_ds.append((data_prep.get_transforms(True)(img), cls))

    aug_loader = DataLoader(aug_ds, batch_size=config.BATCH_SIZE, shuffle=True)

    cand_model = build_model().to(config.DEVICE)
    weights = class_weights(train_items).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.Adam(trainable_parameters(cand_model), lr=config.LEARNING_RATE)

    cand_model.train()
    for epoch in range(5):
        for xb, yb in aug_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            optimiser.zero_grad()
            loss = criterion(cand_model(xb), yb)
            loss.backward()
            optimiser.step()

    # Evaluate candidate on drifted batch
    y_true, y_pred, y_prob = evaluate.predict(cand_model, drift_loader)
    cand_metrics = evaluate.compute_metrics(y_true, y_pred, y_prob)
    cand_f1 = cand_metrics["f1_defect"]

    # Promote vs rollback
    from mlflow import MlflowClient, pytorch
    client = MlflowClient()
    example = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE).numpy()
    pytorch.log_model(cand_model, name=config.REGISTERED_MODEL, input_example=example)
    latest = client.get_latest_versions(config.REGISTERED_MODEL, ["None"])[-1]

    prod_version = client.get_model_version_by_alias(config.REGISTERED_MODEL, config.PRODUCTION_ALIAS)

    if cand_f1 >= prod_f1 + config.PROMOTE_EPSILON:
        action = "promote"
        client.set_registered_model_alias(config.REGISTERED_MODEL, config.PRODUCTION_ALIAS, latest.version)
    else:
        action = "rollback"

    decision = {
        "action": action, "prod_f1": prod_f1, "cand_f1": cand_f1, "prod_metrics": prod_metrics,
        "production_version": prod_version.version, "cand_metrics": cand_metrics,
        "drift_summary": drift, "candidate_version": latest.version
    }

    (config.ARTIFACT_DIR / "retraining_decision.json").write_text(json.dumps(decision, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
