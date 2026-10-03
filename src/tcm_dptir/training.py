from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

from tcm_dptir.datasets import GCADTIDataset, graph_collate_func
from tcm_dptir.models.gcadti import GCADTI


def make_loader(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    max_drug_nodes: int,
    max_protein_len: int,
    use_sampling_weight: bool = True,
):
    dataset = GCADTIDataset(df, max_drug_nodes=max_drug_nodes, max_protein_len=max_protein_len)
    sampler = None
    if shuffle and use_sampling_weight and "sampling_weight" in df.columns:
        weights = torch.as_tensor(df["sampling_weight"].to_numpy(dtype=float), dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=graph_collate_func,
    )


def evaluate(
    model: GCADTI,
    loader: DataLoader,
    device: torch.device,
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[float], list[float]]:
    model.eval()
    labels: list[float] = []
    preds: list[float] = []
    loss_sum = 0.0
    loss_fn = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for drug, protein, label in loader:
            drug = drug.to(device)
            protein = protein.to(device)
            label = label.float().to(device)
            _, logits = model(protein, drug)
            logits = logits.squeeze(1)
            loss = loss_fn(logits, label)
            loss_sum += float(loss.item())
            labels.extend(label.cpu().numpy().tolist())
            preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    metrics = {"loss": loss_sum / max(len(loader), 1)}
    if len(set(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, preds))
        metrics["auprc"] = float(average_precision_score(labels, preds))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["auprc"] = float("nan")
    binary_preds = [int(score >= 0.5) for score in preds]
    metrics["accuracy"] = float(accuracy_score(labels, binary_preds))
    metrics["precision"] = float(precision_score(labels, binary_preds, zero_division=0))
    metrics["recall"] = float(recall_score(labels, binary_preds, zero_division=0))
    metrics["f1"] = float(f1_score(labels, binary_preds, zero_division=0))
    metrics["mcc"] = float(matthews_corrcoef(labels, binary_preds)) if len(set(binary_preds)) > 1 else 0.0
    if len(set(labels)) > 1:
        tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()
        metrics["specificity"] = float(tn / max(tn + fp, 1))
        metrics["sensitivity"] = float(tp / max(tp + fn, 1))
    if return_predictions:
        return metrics, labels, preds
    return metrics


def train_gcadti(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    checkpoint_path: Path,
    test_df: pd.DataFrame | None = None,
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    dropout: float = 0.3,
    weight_decay: float = 0.0,
    optimizer_name: str = "adam",
    label_smoothing: float = 0.0,
    use_sampling_weight: bool = True,
    lr_scheduler_patience: int = 0,
    lr_scheduler_factor: float = 0.5,
    min_learning_rate: float = 1e-6,
    max_drug_nodes: int = 290,
    max_protein_len: int = 1200,
    early_stopping_patience: int = 10,
    seed: int = 42,
    history_path: Path | None = None,
    test_metrics_path: Path | None = None,
    test_predictions_path: Path | None = None,
    checkpoint_metadata_path: Path | None = None,
    training_state_path: Path | None = None,
    resume_training: bool = False,
    initial_checkpoint_path: Path | None = None,
    pairwise_loss_weight: float = 0.0,
    progress_every: int = 0,
    interaction_mode: str = "mutual",
) -> dict[str, object]:
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if not 0.0 <= pairwise_loss_weight <= 1.0:
        raise ValueError("pairwise_loss_weight must be in [0, 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device)
    model = GCADTI(dropout=dropout, interaction_mode=interaction_mode).to(torch_device)
    if initial_checkpoint_path is not None:
        initial_state = torch.load(initial_checkpoint_path, map_location=torch_device)
        if isinstance(initial_state, dict) and "model_state" in initial_state:
            initial_state = initial_state["model_state"]
        model.load_state_dict(initial_state)
        print(f"loaded initial checkpoint={initial_checkpoint_path}", flush=True)
    optimizer_key = optimizer_name.strip().lower()
    if optimizer_key == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_key == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer_name={optimizer_name!r}")
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=min_learning_rate,
        )
        if lr_scheduler_patience > 0
        else None
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_loader = make_loader(
        train_df,
        batch_size,
        True,
        max_drug_nodes,
        max_protein_len,
        use_sampling_weight=use_sampling_weight,
    )
    val_loader = make_loader(val_df, batch_size, False, max_drug_nodes, max_protein_len)
    test_loader = (
        make_loader(test_df, batch_size, False, max_drug_nodes, max_protein_len)
        if test_df is not None
        else None
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    start_epoch = 1

    if resume_training and training_state_path is not None and training_state_path.exists():
        state = torch.load(training_state_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(torch_device)
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        best_auc = float(state["best_auc"])
        best_epoch = int(state["best_epoch"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        history = list(state.get("history", []))
        start_epoch = int(state["epoch"]) + 1
        if state.get("python_random_state") is not None:
            random.setstate(state["python_random_state"])
        if state.get("numpy_random_state") is not None:
            np.random.set_state(state["numpy_random_state"])
        if state.get("torch_random_state") is not None:
            torch.set_rng_state(state["torch_random_state"])
        if torch.cuda.is_available() and state.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_random_state"])
        print(
            f"resume_training state={training_state_path} "
            f"completed_epoch={start_epoch - 1} best_epoch={best_epoch} "
            f"best_val_auc={best_auc:.4f}",
            flush=True,
        )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_started = time.perf_counter()
        for batch_index, (drug, protein, label) in enumerate(train_loader, start=1):
            drug = drug.to(torch_device)
            protein = protein.to(torch_device)
            label = label.float().to(torch_device)
            _, logits = model(protein, drug)
            loss_target = label
            if label_smoothing > 0:
                loss_target = label * (1.0 - label_smoothing) + 0.5 * label_smoothing
            logits_flat = logits.squeeze(1)
            bce_loss = loss_fn(logits_flat, loss_target)
            loss = bce_loss
            if pairwise_loss_weight > 0.0:
                pos = logits_flat[label > 0.5]
                neg = logits_flat[label <= 0.5]
                if pos.numel() > 0 and neg.numel() > 0:
                    pairwise_loss = torch.nn.functional.softplus(
                        -(pos[:, None] - neg[None, :])
                    ).mean()
                    loss = (1.0 - pairwise_loss_weight) * bce_loss + (
                        pairwise_loss_weight * pairwise_loss
                    )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            if progress_every > 0 and (
                batch_index % progress_every == 0 or batch_index == len(train_loader)
            ):
                elapsed = time.perf_counter() - epoch_started
                print(
                    f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"loss={loss.item():.4f} elapsed_s={elapsed:.1f}",
                    flush=True,
                )

        metrics = evaluate(model, val_loader, torch_device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": epoch_loss / max(len(train_loader), 1),
                **{f"val_{key}": value for key, value in metrics.items()},
            }
        )
        print(
            f"epoch={epoch} train_loss={epoch_loss / max(len(train_loader), 1):.4f} "
            f"val_loss={metrics['loss']:.4f} val_auc={metrics['roc_auc']:.4f} "
            f"val_auprc={metrics['auprc']:.4f} lr={current_lr:.2e}",
            flush=True,
        )
        score = metrics["roc_auc"]
        if np.isnan(score):
            score = -metrics["loss"]
        if scheduler is not None:
            scheduler.step(score)
        if score > best_auc:
            best_auc = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        if history_path is not None:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_tmp = history_path.with_suffix(history_path.suffix + ".tmp")
            pd.DataFrame(history).to_csv(history_tmp, index=False)
            history_tmp.replace(history_path)
        if training_state_path is not None:
            training_state_path.parent.mkdir(parents=True, exist_ok=True)
            state_tmp = training_state_path.with_suffix(training_state_path.suffix + ".tmp")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "best_auc": best_auc,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "history": history,
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                    "cuda_random_state": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                },
                state_tmp,
            )
            state_tmp.replace(training_state_path)
        if score <= best_auc:
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"early_stopping epoch={epoch} best_epoch={best_epoch} "
                    f"best_val_auc={best_auc:.4f}",
                    flush=True,
                )
                break

    history_df = pd.DataFrame(history)
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(history_path, index=False)

    result: dict[str, object] = {
        "best_epoch": best_epoch,
        "best_val_roc_auc": float(best_auc),
        "checkpoint": str(checkpoint_path),
        "history": history_df,
    }

    if test_loader is not None and test_df is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch_device))
        test_metrics, test_labels, test_scores = evaluate(
            model, test_loader, torch_device, return_predictions=True
        )
        result["test_metrics"] = test_metrics
        print(
            "test " + " ".join(f"{key}={value:.4f}" for key, value in test_metrics.items()),
            flush=True,
        )

        if test_metrics_path is not None:
            test_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            test_metrics_path.write_text(
                json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if test_predictions_path is not None:
            test_predictions_path.parent.mkdir(parents=True, exist_ok=True)
            predictions = test_df.reset_index(drop=True).copy()
            predictions["observed_label"] = test_labels
            predictions["predicted_score"] = test_scores
            predictions["predicted_label"] = (predictions["predicted_score"] >= 0.5).astype(int)
            predictions.to_csv(test_predictions_path, index=False)

    if checkpoint_metadata_path is not None:
        checkpoint_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "checkpoint": str(checkpoint_path),
            "selection_rule": "maximum validation ROC-AUC; test set not used for model selection",
            "best_epoch": best_epoch,
            "best_val_roc_auc": float(best_auc),
            "seed": seed,
            "epochs_requested": epochs,
            "epochs_completed": len(history_df),
            "early_stopping_patience": early_stopping_patience,
            "batch_size": batch_size,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "optimizer_name": optimizer_key,
            "label_smoothing": label_smoothing,
            "use_sampling_weight": use_sampling_weight,
            "pairwise_loss_weight": pairwise_loss_weight,
            "interaction_mode": interaction_mode,
            "lr_scheduler_patience": lr_scheduler_patience,
            "lr_scheduler_factor": lr_scheduler_factor,
            "min_learning_rate": min_learning_rate,
            "max_drug_nodes": max_drug_nodes,
            "max_protein_len": max_protein_len,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_drug_nodes": max_drug_nodes,
            "max_protein_len": max_protein_len,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df) if test_df is not None else 0,
            "test_metrics": result.get("test_metrics"),
        }
        checkpoint_metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result
