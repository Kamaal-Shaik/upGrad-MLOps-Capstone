"""train.py — Stage 2/3: transfer-learning training + MLflow tracking + registry. 

Implement: build splits, train the ResNet18 head (CrossEntropy, Adam on trainable params,
early stop on val F1), log params/metrics/model to MLflow, register + promote to the
@production alias, evaluate on test, and save a clean-data reference baseline (image
features + embeddings) for drift monitoring.   Run: python -m src.train
"""
from __future__ import annotations

import json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep, evaluate
from src.dataset import CastingDataset
from src.model import build_model, trainable_parameters, save_model, EmbeddingExtractor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _subsample(items, cap):
    # TODO: stratified subsample to `cap` (or return items if cap falsy/too small).
    # raise NotImplementedError

    if not cap or cap <= 0 or cap >= len(items):
        return items
    labels = [cls for _, cls in items]
    _, subsample = train_test_split(items, train_size=cap, stratify=labels, random_state=config.RANDOM_SEED)

    return subsample


def class_weights(items) -> torch.Tensor:
    # TODO: inverse-frequency class weights for CrossEntropyLoss.
    # raise NotImplementedError
    labels = [cls for _, cls in items]
    counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    weights = 1.0 / (counts + 1e-6)  # inverse frequency
    normed = weights / weights.sum() # normalize
    return torch.tensor(normed, dtype=torch.float32)


def save_reference_baseline(net, ref_items) -> dict:
    # TODO 4: save reference_features.csv + reference_embeddings.npz for clean ref images.
    # raise NotImplementedError

    from PIL import Image
    import pandas as pd

    net.eval()
    extractor = EmbeddingExtractor(net) # penultimate-layer embeddings
    tf = data_prep.get_transforms(train=False)

    features, embeddings = [], []
    for img_path, cls in ref_items:
        img = Image.open(img_path)
        feats = data_prep.image_features(img)
        features.append({"path": img_path, "class": config.IDX_TO_CLASS[cls], **feats})

        # embedding vector
        xb = tf(img).unsqueeze(0).to(config.DEVICE)
        emb = extractor(xb)
        embeddings.append(emb.cpu().numpy())

    # Save statistical features
    features_df = pd.DataFrame(features)
    features_csv = config.ARTIFACT_DIR / "reference_features.csv"
    features_df.to_csv(features_csv, index=False)

    # Save embeddings
    embeddings_npz = config.REFERENCE_EMBED
    np.savez(embeddings_npz, embeddings=np.array(embeddings))

    return {"features_csv": str(features_csv), "embeddings_npz": str(embeddings_npz)}



def main() -> int:
    import mlflow, mlflow.pytorch
    from mlflow import MlflowClient
    set_seed()
    root = data_prep.find_data_root()
    qc = data_prep.validate_quality(root)
    (config.ARTIFACT_DIR / "data_quality_report.json").write_text(json.dumps(qc, indent=2))
    data_prep.build_splits(root, "v1")
    # TODO 2/3: load splits, subsample train, build loaders, build model + optimiser + loss.
    # TODO 3 (MLflow): set_experiment; start_run; log_params; per-epoch log_metrics; early stop.
    # TODO 3 (eval + registry): test metrics; plot_eval; save_model; log_model + register +
    #         set @production alias; write model_meta.json + metrics.json.
    # TODO 4: save_reference_baseline on a clean val sample.
    # raise NotImplementedError("Implement the training + MLflow + registry workflow")

    # Load splits
    split_dir = config.SPLIT_DIR / "v1"
    train_items = [(Path(d["path"]), d["label"]) for d in json.load(open(split_dir / "train.json"))["files"]]

    if config.MAX_TRAIN_IMAGES > 0 and len(train_items) >= config.MAX_TRAIN_IMAGES:
        train_items = train_items[:config.MAX_TRAIN_IMAGES]

    val_items = [(Path(d["path"]), d["label"]) for d in json.load(open(split_dir / "val.json"))["files"]]
    test_items = [(Path(d["path"]), d["label"]) for d in json.load(open(split_dir / "test.json"))["files"]]

    # Build datasets/loaders
    train_ds = CastingDataset(train_items, train=True)
    val_ds = CastingDataset(val_items, train=False)
    test_ds = CastingDataset(test_items, train=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # Build model + loss + optimiser
    net = build_model()
    weights = class_weights(train_items).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.Adam(trainable_parameters(net), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # MLflow setup
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": config.EPOCHS, "batch_size": config.BATCH_SIZE, "lr": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY, "freeze_backbone": config.FREEZE_BACKBONE
        })

        best_f1, patience = 0.0, 0
        for epoch in range(config.EPOCHS):
            net.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
                optimiser.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimiser.step()

            # validation
            val_y_true, val_y_pred, val_y_prob = evaluate.predict(net, val_loader)
            val_metrics = evaluate.compute_metrics(val_y_true, val_y_pred, val_y_prob)

            # Only log scalar metrics (skip list/dicts)
            scalar_val_metrics = {f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(scalar_val_metrics, step=epoch)

            # Early stopping on val F1
            if val_metrics["f1_defect"] > best_f1 + 1e-4:
                best_f1 = val_metrics["f1_defect"]
                patience = 0
                save_model(net, config.MODEL_PATH)
            else:
                patience +=1
                if patience >= config.EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Final test evaluation
        test_y_true, test_y_pred, test_y_prob = evaluate.predict(net, test_loader)
        test_metrics = evaluate.compute_metrics(test_y_true, test_y_pred, test_y_prob)

        # Log only scalar metrics
        scalar_test_metrics = {f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(scalar_test_metrics)

        # Save metrics.json
        config.METRICS_PATH.write_text(json.dumps(test_metrics, indent=2))

        # Save evaluation plots
        eval_plot = evaluate.plot_eval(test_y_true, test_y_pred, test_y_prob)
        print("Evaluate plot saved to:", eval_plot)

        # failure cases for report
        failures = evaluate.failure_cases([p for p, _ in test_items], test_y_true, test_y_pred, test_y_prob)
        (config.ARTIFACT_DIR / "failure_cases.json").write_text(json.dumps(failures, indent=2))

        # Register model + promote to @production
        example = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE).numpy()
        mlflow.pytorch.log_model(net, artifact_path="model", input_example=example)
        client = MlflowClient()
        result = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model", name=config.REGISTERED_MODEL
        )
        latest = client.get_latest_versions(config.REGISTERED_MODEL, stages=["None"])[-1]
        client.set_registered_model_alias(config.REGISTERED_MODEL, config.PRODUCTION_ALIAS, latest.version)

        # Save metadata
        meta = {"best_val_f1": best_f1, "test_metrics": test_metrics}
        config.MODEL_META_PATH.write_text(json.dumps(meta, indent=2))

        # Save reference baseline on clean val sample
        ref_items = val_items
        baseline_paths = save_reference_baseline(net, ref_items)
        print("Reference baseline saved:", baseline_paths)


if __name__ == "__main__":
    raise SystemExit(main())
