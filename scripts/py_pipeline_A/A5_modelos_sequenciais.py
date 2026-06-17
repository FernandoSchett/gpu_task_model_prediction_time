#!/usr/bin/env python3
"""Train sequence models (LSTM, GRU and Temporal CNN) for CUDA timing targets."""

from __future__ import annotations

import argparse
import csv
import importlib.util
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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
REGRESSOR_SCRIPT = SCRIPT_DIR / "A2_regressores_classicos.py"
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "resultados" / "analises_sequenciais"


def load_regressor_module():
    spec = importlib.util.spec_from_file_location("regressor_analysis", REGRESSOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Nao foi possivel carregar {REGRESSOR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REG = load_regressor_module()
FEATURES = list(REG.BASE_FEATURES) + [f"kernel_type_{name}" for name in REG.KERNEL_TYPES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--first-sweep", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--sequence-length", type=int, default=int(os.getenv("SEQUENCE_LENGTH", "16")))
    parser.add_argument("--sequence-stride", type=int, default=int(os.getenv("SEQUENCE_STRIDE", "1")))
    parser.add_argument("--max-sequences", type=int, default=int(os.getenv("SEQUENCE_MAX_SEQUENCES", "200000")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("SEQUENCE_EPOCHS", "5")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("SEQUENCE_BATCH_SIZE", "256")))
    parser.add_argument(
        "--split-mode",
        choices=("random", "chronological"),
        default=os.getenv("SEQUENCE_SPLIT_MODE", "random"),
        help="Use random to compare with classical regressors, chronological to test temporal generalization.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("random", "linspace"),
        default=os.getenv("SEQUENCE_SAMPLE_MODE", "random"),
        help="How to downsample sequence windows when --max-sequences is reached.",
    )
    parser.add_argument(
        "--tf-device",
        choices=("cpu", "auto"),
        default=os.getenv("SEQUENCE_TF_DEVICE", "auto"),
        help="Use cpu to avoid TensorFlow CUDA probing, or auto to let TensorFlow choose.",
    )
    parser.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=env_flag("SEQUENCE_REQUIRE_GPU", True),
        help="Fail if --tf-device auto does not expose a TensorFlow GPU.",
    )
    parser.add_argument("--only-model", choices=("lstm", "gru", "temporal_cnn"), default=os.getenv("SEQUENCE_MODEL_ONLY", ""))
    parser.add_argument("--force-model", action="store_true", default=env_flag("SEQUENCE_FORCE_MODEL", False))
    parser.add_argument("--no-cache", action="store_true", help="Recompute even when sequential outputs already exist.")
    return parser.parse_args()


def import_tensorflow(tf_device: str, require_gpu: bool):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if tf_device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow nao esta instalado. Instale com "
            "`python3 -m pip install 'tensorflow[and-cuda]'` "
            "ou rode `bash scripts/configurar_tensorflow_gpu.sh` antes da pipeline A."
        ) from exc
    if tf_device == "auto":
        gpus = tf.config.list_physical_devices("GPU")
        if require_gpu and not gpus:
            raise SystemExit(
                "TensorFlow foi importado, mas nao encontrou GPU. Rode:\n"
                "  bash scripts/configurar_tensorflow_gpu.sh\n"
                "Depois confirme com:\n"
                "  python3 -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"\n"
                "Se quiser rodar temporariamente em CPU, use SEQUENCE_TF_DEVICE=cpu "
                "ou passe --tf-device cpu."
            )
        if gpus:
            print("TensorFlow GPUs:", ", ".join(device.name for device in gpus))
    return tf, keras


def load_jobs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def row_order_key(row: dict[str, str], row_index: int) -> tuple[float, float, float, int]:
    submit_time = REG.to_float(row, "submit_time_ns", math.nan)
    if not math.isfinite(submit_time):
        submit_time = REG.to_float(row, "time_since_experiment_start_us", float(row_index))
    execution_order = REG.to_float(row, "execution_order", math.nan)
    if not math.isfinite(execution_order):
        execution_order = REG.to_float(row, "rank_local_submitted_count", float(row_index))
    completion_time = REG.to_float(row, "completion_time_ns", math.nan)
    if not math.isfinite(completion_time):
        completion_time = submit_time
    return submit_time, execution_order, completion_time, row_index


def load_grouped_events(paths: list[Path], target: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, int]:
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    rows_loaded = 0
    events_used = 0
    for path in paths:
        events: list[tuple[tuple[float, float, float, int], list[float], float]] = []
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "response_time_us" not in reader.fieldnames:
                continue
            for row_index, row in enumerate(reader):
                rows_loaded += 1
                if REG.to_float(row, "cuda_error_code", 0.0) != 0.0:
                    continue
                values = REG.row_features_compare(row)
                y = values[target] if target in values else REG.to_float(row, target)
                x = [values[name] for name in FEATURES]
                if math.isfinite(y) and all(math.isfinite(value) for value in x):
                    events.append((row_order_key(row, row_index), x, y))
        events.sort(key=lambda item: item[0])
        if events:
            groups.append((
                np.asarray([event[1] for event in events], dtype=float),
                np.asarray([event[2] for event in events], dtype=float),
            ))
            events_used += len(events)
    return groups, rows_loaded, events_used


def choose_sequence_specs(
    groups: list[tuple[np.ndarray, np.ndarray]],
    sequence_length: int,
    stride: int,
    max_sequences: int,
    seed: int,
    sample_mode: str,
) -> list[tuple[int, int]]:
    if sequence_length < 2:
        raise SystemExit("--sequence-length precisa ser >= 2.")
    if stride < 1:
        raise SystemExit("--sequence-stride precisa ser >= 1.")
    specs: list[tuple[int, int]] = []
    for group_index, (_x, y) in enumerate(groups):
        count = max(0, (len(y) - sequence_length) // stride)
        specs.extend((group_index, start) for start in range(0, count * stride, stride))
    if max_sequences > 0 and len(specs) > max_sequences:
        indices = np.arange(len(specs))
        if sample_mode == "random":
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(indices, size=max_sequences, replace=False))
        else:
            selected = np.linspace(0, len(specs) - 1, num=max_sequences, dtype=int)
        specs = [specs[index] for index in selected]
    return specs


def make_sequences(
    groups: list[tuple[np.ndarray, np.ndarray]],
    sequence_length: int,
    stride: int,
    max_sequences: int,
    seed: int,
    sample_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_count = groups[0][0].shape[1] if groups else len(FEATURES)
    specs = choose_sequence_specs(groups, sequence_length, stride, max_sequences, seed, sample_mode)
    if not specs:
        return (
            np.empty((0, sequence_length, feature_count), dtype=np.float32),
            np.empty((0, feature_count), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    seq_x = np.empty((len(specs), sequence_length, feature_count), dtype=np.float32)
    current_x = np.empty((len(specs), feature_count), dtype=np.float32)
    seq_y = np.empty((len(specs),), dtype=np.float32)
    for out_index, (group_index, start) in enumerate(specs):
        x, y = groups[group_index]
        end = start + sequence_length
        seq_x[out_index] = x[start:end]
        current_x[out_index] = x[end]
        seq_y[out_index] = y[end]
    return seq_x, current_x, seq_y


def chronological_split(
    history_x: np.ndarray,
    current_x: np.ndarray,
    y: np.ndarray,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < test_fraction < 1.0:
        raise SystemExit("--test-fraction precisa estar entre 0 e 1.")
    test_size = max(1, int(round(len(y) * test_fraction)))
    split = len(y) - test_size
    if split <= 0:
        raise SystemExit("Poucas sequencias para criar treino/teste.")
    return history_x[:split], history_x[split:], current_x[:split], current_x[split:], y[:split], y[split:]


def random_split(
    history_x: np.ndarray,
    current_x: np.ndarray,
    y: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < test_fraction < 1.0:
        raise SystemExit("--test-fraction precisa estar entre 0 e 1.")
    indices = np.arange(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    test_size = max(1, int(round(len(indices) * test_fraction)))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    if len(train_idx) == 0:
        raise SystemExit("Poucas sequencias para criar treino/teste.")
    return (
        history_x[train_idx], history_x[test_idx],
        current_x[train_idx], current_x[test_idx],
        y[train_idx], y[test_idx],
    )


def split_sequences(
    history_x: np.ndarray,
    current_x: np.ndarray,
    y: np.ndarray,
    test_fraction: float,
    seed: int,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split_mode == "random":
        return random_split(history_x, current_x, y, test_fraction, seed)
    return chronological_split(history_x, current_x, y, test_fraction)


def standardize_sequence(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_current_x: np.ndarray,
    test_current_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat_train = np.vstack([train_x.reshape(-1, train_x.shape[-1]), train_current_x])
    mean = flat_train.mean(axis=0)
    scale = flat_train.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (
        (train_x - mean) / scale,
        (test_x - mean) / scale,
        (train_current_x - mean) / scale,
        (test_current_x - mean) / scale,
        mean,
        scale,
    )


def build_models(keras, sequence_length: int, n_features: int) -> dict[str, object]:
    def compile_model(model):
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
        return model

    def hybrid_model(name: str, temporal_layer):
        history_input = keras.layers.Input(shape=(sequence_length, n_features), name="history")
        current_input = keras.layers.Input(shape=(n_features,), name="current")
        temporal = temporal_layer(history_input)
        current = keras.layers.Dense(64, activation="relu")(current_input)
        x = keras.layers.Concatenate()([temporal, current])
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        output = keras.layers.Dense(1)(x)
        return compile_model(keras.Model([history_input, current_input], output, name=name))

    def temporal_cnn_layer(history_input):
        x = keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(history_input)
        x = keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(x)
        return keras.layers.GlobalAveragePooling1D()(x)

    return {
        "lstm": hybrid_model("lstm", keras.layers.LSTM(64)),
        "gru": hybrid_model("gru", keras.layers.GRU(64)),
        "temporal_cnn": hybrid_model("temporal_cnn", temporal_cnn_layer),
    }


def plot_loss(output_dir: Path, model_name: str, history) -> Path:
    path = output_dir / f"{model_name}_training_loss.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history.get("loss", []), label="treino")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="validacao")
    ax.set_title(f"Loss - {model_name}")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "MAE", "RMSE", "R2", "train_sequences", "test_sequences",
        "sequence_length", "sequence_stride", "prediction_horizon", "target_alignment",
        "split_mode", "sample_mode", "diagnostics_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_sequential_metric_comparisons(output_dir: Path, rows: list[dict[str, str]]) -> list[Path]:
    metric_rows: list[dict[str, float | str]] = []
    for row in rows:
        metric_rows.append({
            "model": row["model"],
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "R2": float(row["R2"]),
        })
    return [REG.plot_metric(output_dir, metric_rows, metric) for metric in ("MAE", "RMSE", "R2")]


def expected_metadata(args: argparse.Namespace, job: dict[str, str], paths: list[Path]) -> dict[str, object]:
    return {
        "target": job["target"],
        "label": job["label"],
        "source_files": len(paths),
        "source_paths": [str(path) for path in paths],
        "sequence_grouping": "source_file",
        "sequence_order": "submit_time_ns,execution_order,completion_time_ns,row_index",
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "prediction_horizon": 0,
        "target_alignment": "current_kernel_with_history",
        "input_context": "previous_sequence_plus_current_features",
        "max_sequences": args.max_sequences,
        "test_fraction": args.test_fraction,
        "split_mode": args.split_mode,
        "sample_mode": args.sample_mode,
        "features": FEATURES,
    }


def metadata_matches(existing: dict[str, object], expected: dict[str, object]) -> bool:
    return all(existing.get(key) == value for key, value in expected.items())


def existing_sequential_result(
    job: dict[str, str],
    paths: list[Path],
    seq_output_dir: Path,
    expected: dict[str, object],
) -> dict[str, str] | None:
    metrics_path = seq_output_dir / "sequential_metrics.csv"
    metadata_path = seq_output_dir / "sequence_metadata.json"
    required_models = [seq_output_dir / f"{name}.keras" for name in ("lstm", "gru", "temporal_cnn")]
    required_plots = [
        seq_output_dir / "model_diagnostics" / f"{name}_{suffix}.png"
        for name in ("lstm", "gru", "temporal_cnn")
        for suffix in ("error_distribution", "predicted_vs_actual", "error_qq_normal")
    ]
    comparison_plots = [
        seq_output_dir / "mae_comparison.png",
        seq_output_dir / "rmse_comparison.png",
        seq_output_dir / "r2_comparison.png",
    ]
    if not metrics_path.exists() or not metadata_path.exists():
        return None
    if not all(path.exists() for path in required_models + required_plots):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with metrics_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except (OSError, csv.Error, json.JSONDecodeError):
        return None
    if not metadata_matches(metadata, expected) or not rows:
        return None
    if not all(path.exists() for path in comparison_plots):
        plot_sequential_metric_comparisons(seq_output_dir, rows)
    best = min(rows, key=lambda row: float(row["RMSE"]))
    return {
        "label": job["label"],
        "target": job["target"],
        "source_files": str(len(paths)),
        "rows_loaded": str(metadata.get("rows_loaded", "")),
        "train_sequences": str(metadata.get("train_sequences", "")),
        "test_sequences": str(metadata.get("test_sequences", "")),
        "best_model": best["model"],
        "best_rmse": best["RMSE"],
        "best_r2": best["R2"],
        "metrics_csv": str(metrics_path),
        "output_dir": str(seq_output_dir),
        "cached": "true",
    }


def run_job(args: argparse.Namespace, job: dict[str, str], keras) -> dict[str, str]:
    paths = REG.result_paths(args.results_dir, args.first_sweep, job["include_regex"])
    output_dir = Path(job["output_dir"])
    seq_output_dir = output_dir / "sequenciais"
    diagnostics_dir = seq_output_dir / "model_diagnostics"
    seq_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_signature = expected_metadata(args, job, paths)
    metadata_path = seq_output_dir / "sequence_metadata.json"
    reusable_model_cache = False
    if metadata_path.exists():
        try:
            reusable_model_cache = metadata_matches(
                json.loads(metadata_path.read_text(encoding="utf-8")),
                metadata_signature,
            )
        except (OSError, json.JSONDecodeError):
            reusable_model_cache = False

    if not args.no_cache and not args.only_model and not args.force_model:
        cached = existing_sequential_result(job, paths, seq_output_dir, metadata_signature)
        if cached is not None:
            print(f"{job['label']} {job['target']}: cached sequenciais={seq_output_dir}")
            return cached

    groups, rows_loaded, events_used = load_grouped_events(paths, job["target"])
    seq_x, current_x, seq_y = make_sequences(
        groups,
        args.sequence_length,
        args.sequence_stride,
        args.max_sequences,
        args.seed,
        args.sample_mode,
    )
    if len(seq_y) == 0:
        raise SystemExit(f"Nenhuma sequencia valida para {job['label']} {job['target']}.")
    train_x, test_x, train_current_x, test_current_x, train_y, test_y = split_sequences(
        seq_x,
        current_x,
        seq_y,
        args.test_fraction,
        args.seed,
        args.split_mode,
    )
    train_x, test_x, train_current_x, test_current_x, mean, scale = standardize_sequence(
        train_x,
        test_x,
        train_current_x,
        test_current_x,
    )

    metadata = {
        **metadata_signature,
        "rows_loaded": int(rows_loaded),
        "events_used": int(events_used),
        "groups": int(len(groups)),
        "sequences": int(len(seq_y)),
        "train_sequences": int(len(train_y)),
        "test_sequences": int(len(test_y)),
        "standardize": {"mean": mean.tolist(), "scale": scale.tolist()},
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, str]] = []
    residual_metrics_path = diagnostics_dir / "residual_distribution_metrics.csv"
    if residual_metrics_path.exists():
        residual_metrics_path.unlink()
    for model_name, model in build_models(keras, args.sequence_length, train_x.shape[-1]).items():
        if args.only_model and model_name != args.only_model:
            continue
        model_path = seq_output_dir / f"{model_name}.keras"
        model_inputs = [test_x, test_current_x]
        if reusable_model_cache and model_path.exists() and not args.force_model and not args.no_cache:
            try:
                model = keras.models.load_model(model_path)
                prediction = model.predict(model_inputs, batch_size=args.batch_size, verbose=0).reshape(-1)
                history = None
            except Exception as exc:
                print(f"{job['label']} {job['target']} {model_name}: cache incompativel, retreinando ({exc})")
                model = build_models(keras, args.sequence_length, train_x.shape[-1])[model_name]
                history = model.fit(
                    [train_x, train_current_x],
                    train_y,
                    validation_split=0.1,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    verbose=0,
                    shuffle=False,
                )
                prediction = model.predict(model_inputs, batch_size=args.batch_size, verbose=0).reshape(-1)
        else:
            history = model.fit(
                [train_x, train_current_x],
                train_y,
                validation_split=0.1,
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=0,
                shuffle=False,
            )
            prediction = model.predict(model_inputs, batch_size=args.batch_size, verbose=0).reshape(-1)
        metric_row = REG.metrics(test_y, prediction)
        REG.plot_prediction_diagnostics(seq_output_dir, model_name, test_y, prediction)
        if history is not None:
            plot_loss(seq_output_dir, model_name, history)
            model.save(model_path)
        rows.append({
            "model": model_name,
            "MAE": f"{metric_row['MAE']:.6f}",
            "RMSE": f"{metric_row['RMSE']:.6f}",
            "R2": f"{metric_row['R2']:.6f}",
            "train_sequences": str(len(train_y)),
            "test_sequences": str(len(test_y)),
            "sequence_length": str(args.sequence_length),
            "sequence_stride": str(args.sequence_stride),
            "prediction_horizon": "0",
            "target_alignment": "current_kernel_with_history",
            "split_mode": args.split_mode,
            "sample_mode": args.sample_mode,
            "diagnostics_dir": str(diagnostics_dir),
        })

    if not rows:
        raise SystemExit(f"Modelo sequencial nao encontrado: {args.only_model}")
    metrics_path = seq_output_dir / "sequential_metrics.csv"
    write_metrics(metrics_path, rows)
    plot_sequential_metric_comparisons(seq_output_dir, rows)
    best = min(rows, key=lambda row: float(row["RMSE"]))
    return {
        "label": job["label"],
        "target": job["target"],
        "source_files": str(len(paths)),
        "rows_loaded": str(rows_loaded),
        "train_sequences": rows[0]["train_sequences"],
        "test_sequences": rows[0]["test_sequences"],
        "best_model": best["model"],
        "best_rmse": best["RMSE"],
        "best_r2": best["R2"],
        "metrics_csv": str(metrics_path),
        "output_dir": str(seq_output_dir),
        "cached": "false",
    }


def main() -> int:
    args = parse_args()
    tf, keras = import_tensorflow(args.tf_device, args.require_gpu)
    tf.keras.utils.set_random_seed(args.seed)
    rows = [run_job(args, job, keras) for job in load_jobs(args.jobs_file)]
    summary_path = args.analysis_dir / "sequential_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "label", "target", "source_files", "rows_loaded", "train_sequences",
            "test_sequences", "best_model", "best_rmse", "best_r2", "metrics_csv", "output_dir", "cached",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"sequential_summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
