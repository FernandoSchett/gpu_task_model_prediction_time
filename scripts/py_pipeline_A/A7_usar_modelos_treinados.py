#!/usr/bin/env python3
"""Example: How to load and reuse trained models for predictions on new data."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow not available, CNN predictions will be skipped")


def load_trained_models(models_dir: Path) -> dict:
    """Load all available trained models from directory."""
    models = {}
    
    # Load sklearn models
    sklearn_models = ["linear_regression", "ridge_regression", "polynomial_ridge", 
                     "decision_tree", "random_forest", "gradient_boosting"]
    
    for model_name in sklearn_models:
        model_path = models_dir / f"{model_name}.pkl"
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    models[model_name] = pickle.load(f)
                print(f"✅ Loaded: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
    
    # Load CNN model if available
    if KERAS_AVAILABLE:
        cnn_path = models_dir / "cnn_model.h5"
        if cnn_path.exists():
            try:
                models["cnn"] = keras.models.load_model(str(cnn_path))
                print(f"✅ Loaded: CNN Neural Network")
            except Exception as e:
                print(f"❌ Failed to load CNN: {e}")
    
    return models


def make_predictions(models: dict, X_normalized: np.ndarray, X_original: np.ndarray = None) -> dict:
    """Make predictions using all available models."""
    predictions = {}
    
    # Sklearn models predictions
    for model_name, model in models.items():
        if model_name == "cnn":
            continue
        
        try:
            # Handle polynomial ridge which uses quadratic features
            if model_name == "polynomial_ridge":
                X_quad = X_normalized * X_normalized
                X_input = np.column_stack([X_normalized, X_quad])
            else:
                X_input = X_normalized
            
            predictions[model_name] = model.predict(X_input)
        except Exception as e:
            print(f"⚠️  Error predicting with {model_name}: {e}")
    
    # CNN predictions
    if "cnn" in models:
        try:
            X_reshaped = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], 1)
            cnn_preds = models["cnn"].predict(X_reshaped, verbose=0).flatten()
            predictions["cnn"] = cnn_preds
        except Exception as e:
            print(f"⚠️  Error predicting with CNN: {e}")
    
    return predictions


def main():
    """Example usage."""
    # 1. Define paths
    models_dir = Path("resultados/analises_regressao") / "seu_sweep" / "geral" / "response_time_us" / "trained_models"
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print("Please run the analysis first:")
        print("  python3 scripts/py_pipeline_A/A2_regressores_classicos.py compare --results-dir ./resultados/seu_sweep")
        return 1
    
    # 2. Load trained models
    print(f"Loading models from: {models_dir}\n")
    models = load_trained_models(models_dir)
    
    if not models:
        print("❌ No models found!")
        return 1
    
    # 3. Prepare example data (replace with your actual data)
    print("\n📊 Example: Making predictions on synthetic data")
    print("-" * 50)
    
    # Replace this with your actual data loading
    X_new = np.random.randn(100, 24)  # 100 samples, 24 features
    
    # 4. Normalize data (IMPORTANT: use same normalization as training)
    # In production, you should save and load the training scaler
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_new)
    
    # 5. Make predictions
    predictions = make_predictions(models, X_normalized, X_new)
    
    # 6. Display results
    print(f"\n✅ Predictions completed for {len(predictions)} models\n")
    
    for model_name, preds in predictions.items():
        print(f"{model_name:20} | Mean: {preds.mean():.3f} | Std: {preds.std():.3f} | Min: {preds.min():.3f} | Max: {preds.max():.3f}")
    
    # 7. Compare predictions
    if len(predictions) > 1:
        print(f"\n📈 Model Agreement (std across model predictions):")
        print("-" * 50)
        model_preds_array = np.array(list(predictions.values()))
        prediction_std = model_preds_array.std(axis=0)
        print(f"Mean agreement std: {prediction_std.mean():.3f}")
        print(f"High uncertainty samples (std > 0.5σ): {(prediction_std > prediction_std.mean() + 0.5*prediction_std.std()).sum()}")
    
    # 8. Save predictions
    output_csv = Path("new_predictions.csv")
    print(f"\n💾 Saving predictions to: {output_csv}")
    
    # Simple CSV output (you can make this more sophisticated)
    predictions["sample_id"] = np.arange(len(X_new))
    df_data = {k: v for k, v in predictions.items()}
    
    try:
        import pandas as pd
        df = pd.DataFrame(df_data)
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved {len(df)} predictions")
    except ImportError:
        print("⚠️  pandas not available, skipping CSV output")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
