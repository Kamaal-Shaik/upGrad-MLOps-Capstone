"""evaluate.py — Stage 3: evaluation metrics, plots, failure-case analysis. 

Positive class = DEFECT (recall on defects is the headline QC metric). Implement
prediction, imbalance-aware metrics, confusion/ROC plots, and a misclassified-sample list.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

from src.model import EmbeddingExtractor
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score, roc_auc_score, consensus_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt

@torch.no_grad()
def predict(net, loader, return_embeddings: bool = False):
    # TODO 3: return (y_true, y_pred, y_prob_defect[, embeddings]) over the loader.
    # raise NotImplementedError
    y_true, y_pred, y_prob, embeddings = [], [], [], []
    net.eval()
    for xb, yb in loader:
        xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
        logits = net(xb)
        probs = F.softmax(logits, dim=1)[:, config.POSITIVE_IDX] # detect prob
        preds = torch.argmax(logits, dim=1)

        y_true.extend(yb.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

        if return_embeddings:
            extractor = EmbeddingExtractor(net)
            emb = extractor(xb)
            embeddings.extend(emb.cpu().numpy())

    if return_embeddings:
        return np.array(y_true), np.array(y_pred), np.array(y_prob), np.array(embeddings)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    # TODO 3: accuracy, precision/recall/f1 for the DEFECT class (pos_label=config.POSITIVE_IDX),
    #         macro_f1, roc_auc, confusion_matrix. Return a dict.
    # raise NotImplementedError
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=config.POSITIVE_IDX, average="binary", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    roc_auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": acc, "precision_defect": prec, "recall_defect": rec, "f1_defect": f1,
        "macro_f1": macro_f1, "roc_auc": roc_auc, "confusion_matrix": cm
    }

def plot_eval(y_true, y_pred, y_prob, out: Path | None = None) -> Path:
    # TODO 3: save a confusion-matrix + ROC figure to artifacts/model_eval.png.
    # raise NotImplementedError
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, cm[i,j], ha="center", va="center")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[1].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.2f}")
    axes[1].plot([0,1], [0,1], "--", color="gray")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    out = out or (config.ARTIFACT_DIR / "model_eval.png")
    fig.savefig(out)
    return out


def failure_cases(items, y_true, y_pred, y_prob, limit: int = 20) -> list[dict]:
    # TODO 3: list misclassified samples with predicted p_defect (error analysis).
    # raise NotImplementedError
    errors = []
    for path, true, pred, prob in zip(items, y_true, y_pred, y_prob):
        if true != pred:
            errors.append({
                "path": str(path), "true": config.IDX_TO_CLASS[true],
                "pred": config.IDX_TO_CLASS[pred], "p_defect": float(prob)
            })

    return errors[:limit]
