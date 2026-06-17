#!/usr/bin/env python3
"""Train and compare 2D CNN architectures for Pipeline C."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "sim"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--preprocess-summary", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=int(os.getenv("CNN2D_EPOCHS", "8")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("CNN2D_BATCH_SIZE", "128")))
    parser.add_argument("--test-fraction", type=float, default=float(os.getenv("CNN2D_TEST_FRACTION", "0.25")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42") or "42"))
    parser.add_argument("--max-architectures", type=int, default=int(os.getenv("CNN2D_MAX_ARCHITECTURES", "8")))
    parser.add_argument("--tf-device", choices=("auto", "cpu"), default=os.getenv("CNN2D_TF_DEVICE", "auto"))
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=env_flag("CNN2D_REQUIRE_GPU", True))
    parser.add_argument("--only-model", default=os.getenv("CNN2D_MODEL_ONLY", ""), help="Train only one 2D CNN architecture by name.")
    parser.add_argument("--force-model", action="store_true", default=env_flag("CNN2D_FORCE_MODEL", False), help="Retrain selected architecture even when cached model exists.")
    parser.add_argument("--plots-only", action="store_true", default=env_flag("CNN2D_PLOTS_ONLY", False), help="Regenerate prediction/error plots from cached 2D models without training.")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def import_tensorflow(tf_device: str, require_gpu: bool):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if tf_device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit("TensorFlow nao esta instalado no ambiente Python da Pipeline C.") from exc
    if tf_device == "auto":
        gpus = tf.config.list_physical_devices("GPU")
        if require_gpu and not gpus:
            raise SystemExit("TensorFlow nao encontrou GPU para a Pipeline C.")
        if gpus:
            print("TensorFlow GPUs:", ", ".join(device.name for device in gpus))
    return tf, keras


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = math.sqrt(float(np.mean(residual * residual)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual * residual)) / denom if denom > 0 else math.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_test_split(x: np.ndarray, y: np.ndarray, test_fraction: float, seed: int):
    indices = np.arange(len(y), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    test_size = max(1, int(round(len(indices) * test_fraction)))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    if len(train_idx) == 0:
        raise SystemExit("Poucas amostras para treino/teste na Pipeline C.")
    return train_idx, test_idx


def split_train_validation(train_idx: np.ndarray, seed: int, validation_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    if len(train_idx) < 2:
        raise SystemExit("Poucas amostras para separar validacao.")
    rng = np.random.default_rng(seed + 1)
    shuffled = np.array(train_idx, copy=True)
    rng.shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * validation_fraction)))
    return shuffled[val_size:], shuffled[:val_size]


def standardization_stats(x: np.ndarray, indices: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    total = 0
    sum_values: np.ndarray | None = None
    sumsq_values: np.ndarray | None = None
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = np.asarray(x[batch_idx], dtype=np.float32)
        axes = (0, 1, 2)
        batch_sum = batch.sum(axis=axes, keepdims=True, dtype=np.float64)
        batch64 = batch.astype(np.float64, copy=False)
        batch_sumsq = np.square(batch64).sum(axis=axes, keepdims=True)
        count = batch.shape[0] * batch.shape[1] * batch.shape[2]
        if sum_values is None:
            sum_values = batch_sum
            sumsq_values = batch_sumsq
        else:
            sum_values += batch_sum
            sumsq_values += batch_sumsq
        total += count
    if total == 0 or sum_values is None or sumsq_values is None:
        raise SystemExit("Nenhuma amostra para normalizacao CNN2D.")
    mean = sum_values / float(total)
    variance = np.maximum((sumsq_values / float(total)) - (mean * mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0.0] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


class MemmapSequence:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        batch_size: int,
        mean: np.ndarray,
        scale: np.ndarray,
        shuffle: bool,
        seed: int,
    ):
        self.x = x
        self.y = y
        self.indices = np.array(indices, dtype=np.int64, copy=True)
        self.batch_size = batch_size
        self.mean = mean
        self.scale = scale
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index: int):
        batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = np.asarray(self.x[batch_idx], dtype=np.float32)
        batch_x = (batch_x - self.mean) / self.scale
        batch_y = np.asarray(self.y[batch_idx], dtype=np.float32)
        return batch_x, batch_y

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)


def architecture_grid(max_architectures: int) -> list[dict[str, float | int | str]]:
    grid: list[dict[str, float | int | str]] = []
    for filters in (16, 32):
        for temporal_kernel in (3, 5):
            for dense_units in (64, 128):
                grid.append({
                    "name": f"cnn2d_f{filters}_kt{temporal_kernel}_d{dense_units}",
                    "filters": filters,
                    "temporal_kernel": temporal_kernel,
                    "dense_units": dense_units,
                    "dropout": 0.10,
                    "learning_rate": 0.001,
                    "extra_block": 0,
                })
                grid.append({
                    "name": f"cnn2d_deep_f{filters}_kt{temporal_kernel}_d{dense_units}",
                    "filters": filters,
                    "temporal_kernel": temporal_kernel,
                    "dense_units": dense_units,
                    "dropout": 0.20,
                    "learning_rate": 0.001,
                    "extra_block": 1,
                })
    return grid[:max_architectures] if max_architectures > 0 else grid


def selected_architectures(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    grid = architecture_grid(args.max_architectures)
    if not args.only_model:
        return grid
    selected = [params for params in grid if str(params["name"]) == args.only_model]
    if selected:
        return selected
    known = ", ".join(str(params["name"]) for params in grid)
    raise SystemExit(f"Arquitetura CNN2D nao encontrada: {args.only_model}. Opcoes: {known}")


def build_model(keras, input_shape: tuple[int, int, int], params: dict[str, float | int | str]):
    workers, window_size, _features = input_shape
    temporal_kernel = min(int(params["temporal_kernel"]), window_size)
    filters = int(params["filters"])
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(
        filters,
        kernel_size=(workers, temporal_kernel),
        padding="valid",
        activation="relu",
        name="all_workers_temporal_conv",
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    if int(params["extra_block"]):
        x = keras.layers.Conv2D(filters * 2, kernel_size=(1, 3), padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(float(params["dropout"]))(x)
    x = keras.layers.Dense(int(params["dense_units"]), activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=str(params["name"]))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(params["learning_rate"])),
        loss="mse",
        metrics=["mae"],
    )
    return model


def plot_prediction(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = min(len(y_true), 5000)
    rng = np.random.default_rng(42)
    idx = np.arange(len(y_true)) if len(y_true) <= sample else np.sort(rng.choice(len(y_true), sample, replace=False))
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(y_true[idx], y_pred[idx], s=7, alpha=0.35)
    lo = float(min(y_true[idx].min(), y_pred[idx].min()))
    hi = float(max(y_true[idx].max(), y_pred[idx].max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Real")
    ax.set_ylabel("Predito")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_distribution(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residual = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residual, bins=80, color="#4c78a8", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Erro")
    ax.set_ylabel("Frequencia")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss(path: Path, history) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history.get("loss", []), label="train")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residual = y_true - y_pred
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "y_true", "y_pred", "error", "abs_error"])
        writer.writeheader()
        for index, (true_value, pred_value, error) in enumerate(zip(y_true, y_pred, residual)):
            writer.writerow({
                "index": index,
                "y_true": f"{float(true_value):.9g}",
                "y_pred": f"{float(pred_value):.9g}",
                "error": f"{float(error):.9g}",
                "abs_error": f"{abs(float(error)):.9g}",
            })


def existing_metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def merge_metric_rows(old_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for row in old_rows:
        architecture = row.get("architecture", "")
        if architecture:
            merged[architecture] = row
    for row in new_rows:
        merged[row["architecture"]] = row
    return sorted(merged.values(), key=lambda row: (-float(row["R2"]), float(row["RMSE"]), row["architecture"]))


def load_dataset(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, Path]:
    tensor_path = Path(row["tensor_path"])
    y_path_value = row.get("y_path", "")
    if y_path_value:
        x = np.load(tensor_path, mmap_mode="r")
        y = np.load(Path(y_path_value), mmap_mode="r")
        return x, y, tensor_path.parent
    data = np.load(tensor_path)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    return x, y, tensor_path.parent


def run_dataset(args: argparse.Namespace, row: dict[str, str], keras) -> list[dict[str, str]]:
    x, y, output_dir = load_dataset(row)
    models_dir = output_dir / "trained_models"
    plots_dir = output_dir / "model_diagnostics"
    predictions_dir = output_dir / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    train_idx, test_idx = train_test_split(x, y, args.test_fraction, args.seed)
    fit_idx, val_idx = split_train_validation(train_idx, args.seed)
    mean, scale = standardization_stats(x, fit_idx, args.batch_size)

    class KerasMemmapSequence(MemmapSequence, keras.utils.Sequence):
        pass

    train_seq = KerasMemmapSequence(x, y, fit_idx, args.batch_size, mean, scale, True, args.seed)
    val_seq = KerasMemmapSequence(x, y, val_idx, args.batch_size, mean, scale, False, args.seed)
    test_seq = KerasMemmapSequence(x, y, test_idx, args.batch_size, mean, scale, False, args.seed)
    test_y = np.asarray(y[test_idx], dtype=np.float32)

    preprocessing_path = models_dir / "cnn2d_preprocessing.npz"
    np.savez_compressed(preprocessing_path, mean=mean, scale=scale)

    results: list[dict[str, str]] = []
    for params in selected_architectures(args):
        name = str(params["name"])
        model_path = models_dir / f"{name}.keras"
        metadata_path = models_dir / f"{name}.json"
        metrics_path = models_dir / f"{name}_metrics.json"
        cached = False

        use_cache = not args.no_cache and not args.force_model
        if use_cache and model_path.exists():
            model = keras.models.load_model(model_path)
            prediction = model.predict(test_seq, verbose=0).reshape(-1)
            metric_row = metrics(test_y, prediction)
            cached = True
            if not metrics_path.exists():
                metrics_path.write_text(json.dumps(metric_row, indent=2) + "\n", encoding="utf-8")
        else:
            if args.plots_only:
                raise SystemExit(
                    f"Modelo CNN2D ausente para plots-only: {model_path}. "
                    "Rode sem CNN2D_PLOTS_ONLY ou treine esta arquitetura primeiro."
                )
            model = build_model(keras, tuple(x.shape[1:]), params)
            history = model.fit(
                train_seq,
                validation_data=val_seq,
                epochs=args.epochs,
                verbose=0,
            )
            prediction = model.predict(test_seq, verbose=0).reshape(-1)
            metric_row = metrics(test_y, prediction)
            model.save(model_path)
            metadata_path.write_text(
                json.dumps(
                    {
                        "architecture": params,
                        "input_shape": list(x.shape[1:]),
                        "target_alignment": "next_kernel",
                        "preprocessing_path": str(preprocessing_path),
                    },
                    indent=2,
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
            metrics_path.write_text(json.dumps(metric_row, indent=2) + "\n", encoding="utf-8")
            plot_loss(plots_dir / f"{name}_loss.png", history)

        plot_prediction(plots_dir / f"{name}_predicted_vs_actual.png", test_y, prediction, name)
        plot_error_distribution(plots_dir / f"{name}_error_distribution.png", test_y, prediction, name)
        write_predictions(predictions_dir / f"{name}_predictions.csv", test_y, prediction)
        results.append({
            "label": row["label"],
            "target": row["target"],
            "architecture": name,
            "MAE": f"{metric_row['MAE']:.6f}",
            "RMSE": f"{metric_row['RMSE']:.6f}",
            "R2": f"{metric_row['R2']:.6f}",
            "train_samples": str(len(fit_idx)),
            "test_samples": str(len(test_y)),
            "workers": row["workers"],
            "window_size": row["window_size"],
            "features": row["features"],
            "model_path": str(model_path),
            "plots_dir": str(plots_dir),
            "predictions_csv": str(predictions_dir / f"{name}_predictions.csv"),
            "cached": "true" if cached else "false",
        })
        print(
            f"cnn2d_train {row['label']} {row['target']} {name}: "
            f"rmse={metric_row['RMSE']:.3f} r2={metric_row['R2']:.3f}"
            f"{' cached' if cached else ''}"
        )
    metrics_csv = output_dir / "cnn2d_architecture_metrics.csv"
    if args.only_model:
        results = merge_metric_rows(existing_metric_rows(metrics_csv), results)
    write_metrics(metrics_csv, results)
    return results


def write_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label", "target", "architecture", "MAE", "RMSE", "R2", "train_samples",
        "test_samples", "workers", "window_size", "features", "model_path", "plots_dir", "cached",
        "predictions_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    tf, keras = import_tensorflow(args.tf_device, args.require_gpu)
    tf.keras.utils.set_random_seed(args.seed)
    all_rows: list[dict[str, str]] = []
    for row in load_rows(args.preprocess_summary):
        all_rows.extend(run_dataset(args, row, keras))
    summary_path = args.analysis_dir / "pipeline_c_cnn2d_training_summary.csv"
    write_metrics(summary_path, all_rows)
    print(f"pipeline_c_cnn2d_training_summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
