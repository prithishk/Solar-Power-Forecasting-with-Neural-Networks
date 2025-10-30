# =====================================================
# Solar Power Forecasting with Neural Networks
# Cleaned and stabilized for macOS / TensorFlow
# =====================================================

import os
import argparse
import random
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras

# -----------------------
# System Config Fixes
# -----------------------
# Suppress TensorFlow C++ logs and lock warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Limit threads (macOS + Metal backend stability)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# -----------------------
# Config / defaults
# -----------------------
DEFAULT_WEATHER_CSV = "/Users/admin/Desktop/GIT/Prithish/Solar_Power_Forecasting_with_Neural_Networks/Plant_1_Weather_Sensor_Data.csv"
DEFAULT_GEN_CSV     = "/Users/admin/Desktop/GIT/Prithish/Solar_Power_Forecasting_with_Neural_Networks/Plant_1_Generation_Data.csv"

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------
# Helper Functions
# -----------------------
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def build_ffnn(input_dim, hidden_layers=(64,64), activation="relu",
               dropout=0.0, batch_norm=False, l2=0.0, lr=1e-3):
    """Build Feed-Forward Neural Network"""
    model = keras.Sequential()
    for i, units in enumerate(hidden_layers):
        if i == 0:
            model.add(keras.layers.Dense(units, activation=activation,
                                         kernel_regularizer=keras.regularizers.l2(l2) if l2 > 0 else None,
                                         input_shape=(input_dim,)))
        else:
            model.add(keras.layers.Dense(units, activation=activation,
                                         kernel_regularizer=keras.regularizers.l2(l2) if l2 > 0 else None))
        if batch_norm:
            model.add(keras.layers.BatchNormalization())
        if dropout and dropout > 0:
            model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation="linear"))
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def print_metrics_dict(m):
    print(f"MAE: {m['MAE']:.4f} | RMSE: {m['RMSE']:.4f} | R2: {m['R2']:.4f}")

# -----------------------
# Main pipeline
# -----------------------
def run_pipeline(weather_csv, gen_csv,
                 features=['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION'],
                 target='AC_POWER',
                 resample_freq='H',
                 irradiation_threshold=None,
                 chronological_split=True,
                 split_ratio=0.8,
                 baseline_epochs=50,
                 ffnn_cfg=dict(hidden_layers=(128,64), activation="relu", dropout=0.1,
                               batch_norm=True, l2=1e-5, lr=1e-3),
                 fit_cfg=dict(epochs=150, batch_size=64),
                 save_model_path=None):
    set_seeds(SEED)

    # 1) Load data
    print("📥 Loading CSVs...")
    weather_df = pd.read_csv(weather_csv)
    generation_df = pd.read_csv(gen_csv)

    print("Weather columns:", weather_df.columns.tolist())
    print("Generation columns:", generation_df.columns.tolist())

    # 2) Convert datetime and sort
    print("🕒 Parsing DATE_TIME...")
    if 'DATE_TIME' not in weather_df.columns or 'DATE_TIME' not in generation_df.columns:
        raise ValueError("Both CSVs must contain 'DATE_TIME' column.")

    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], dayfirst=True, errors='coerce')
    generation_df['DATE_TIME'] = pd.to_datetime(generation_df['DATE_TIME'], dayfirst=True, errors='coerce')

    weather_df = weather_df.dropna(subset=['DATE_TIME']).set_index('DATE_TIME').sort_index()
    generation_df = generation_df.dropna(subset=['DATE_TIME']).set_index('DATE_TIME').sort_index()

    # 3) Resample and aggregate
    print(f"🔁 Resampling ({resample_freq})...")
    if 'SOURCE_KEY' in generation_df.columns:
        numeric_cols = generation_df.select_dtypes(include=[np.number]).columns.tolist()
        gen_grouped = (generation_df.reset_index()
                       .groupby(['SOURCE_KEY', pd.Grouper(key='DATE_TIME', freq=resample_freq)])[numeric_cols]
                       .sum().reset_index()
                       .groupby('DATE_TIME')[numeric_cols].sum())
        generation_res = gen_grouped.sort_index()
    else:
        numeric_cols = generation_df.select_dtypes(include=[np.number]).columns.tolist()
        generation_res = generation_df[numeric_cols].resample(resample_freq).sum()

    weather_drop = weather_df.copy()
    for c in ['PLANT_ID','SOURCE_KEY']:
        if c in weather_drop.columns:
            weather_drop = weather_drop.drop(columns=[c])
    weather_res = weather_drop.resample(resample_freq).mean()

    # 4) Merge dataframes
    print("🔗 Aligning generation & weather data...")
    master_index = weather_res.index
    generation_res = generation_res.reindex(master_index).fillna(0)
    weather_res = weather_res.reindex(master_index)

    df = pd.concat([
        generation_res[[target]] if target in generation_res.columns else
        generation_res.iloc[:, 0:1].rename(columns={generation_res.columns[0]: target}),
        weather_res[features]
    ], axis=1)

    df = df.reset_index().rename(columns={'index': 'DATE_TIME'})
    df['DATE'] = pd.to_datetime(df['DATE_TIME']).dt.date
    df = df.dropna(subset=features + [target])
    print(f"✅ Rows after cleaning: {len(df)}")

    # 5) Daytime filter
    if irradiation_threshold is not None and 'IRRADIATION' in df.columns:
        before = len(df)
        df = df[df['IRRADIATION'] >= irradiation_threshold]
        print(f"☀️ Filtered daylight (IRRADIATION ≥ {irradiation_threshold}): {before} → {len(df)}")

    # 6) Chronological split
    df = df.sort_values('DATE_TIME').reset_index(drop=True)
    unique_dates = np.array(sorted(df['DATE'].unique()))
    split_index = int(len(unique_dates) * split_ratio)
    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]

    train_df = df[df['DATE'].isin(train_dates)]
    test_df = df[df['DATE'].isin(test_dates)]

    print(f"🧩 Train: {len(train_df)} | Test: {len(test_df)}")

    # 7) Scale data
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train = X_scaler.fit_transform(train_df[features])
    y_train = y_scaler.fit_transform(train_df[[target]]).ravel()
    X_test = X_scaler.transform(test_df[features])
    y_test = y_scaler.transform(test_df[[target]]).ravel()

    print("🔢 Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # 8) Baseline model
    print("\n⚙️ Training Baseline Linear Model...")
    baseline = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(1, activation='linear')
    ])
    baseline.compile(optimizer='adam', loss='mse', metrics=['mae'])
    baseline.fit(X_train, y_train.reshape(-1,1), epochs=baseline_epochs, validation_split=0.1, verbose=1)

    # Evaluate baseline
    y_pred_base = baseline.predict(X_test).ravel()
    y_pred_base_inv = y_scaler.inverse_transform(y_pred_base.reshape(-1,1)).ravel()
    y_true_inv = y_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
    base_metrics = metrics(y_true_inv, y_pred_base_inv)
    print("\n📊 Baseline Performance:")
    print_metrics_dict(base_metrics)

    # 9) FFNN
    print("\n⚙️ Training FFNN...")
    ffnn = build_ffnn(X_train.shape[1], **ffnn_cfg)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=7, factor=0.5, min_lr=1e-6)
    ]
    ffnn.fit(X_train, y_train.reshape(-1,1), validation_split=0.1,
             epochs=fit_cfg.get('epochs',150),
             batch_size=fit_cfg.get('batch_size',64),
             callbacks=callbacks, verbose=1, shuffle=True)

    y_pred_ffnn = ffnn.predict(X_test).ravel()
    y_pred_ffnn_inv = y_scaler.inverse_transform(y_pred_ffnn.reshape(-1,1)).ravel()
    ffnn_metrics = metrics(y_true_inv, y_pred_ffnn_inv)
    print("\n📊 FFNN Performance:")
    print_metrics_dict(ffnn_metrics)

    # 10) Compare
    print("\n🔍 Comparison Table:")
    comp = pd.DataFrame({
        "Model": ["Baseline", "FFNN"],
        "MAE": [base_metrics["MAE"], ffnn_metrics["MAE"]],
        "RMSE": [base_metrics["RMSE"], ffnn_metrics["RMSE"]],
        "R2": [base_metrics["R2"], ffnn_metrics["R2"]]
    })
    print(comp.to_string(index=False, float_format="%.4f"))

    # 11) Plot
    results = test_df[['DATE_TIME']].copy()
    results['True'] = y_true_inv
    results['Baseline'] = y_pred_base_inv
    results['FFNN'] = y_pred_ffnn_inv

    plt.figure(figsize=(14,5))
    plt.plot(results['DATE_TIME'][:1000], results['True'][:1000], label="True", linewidth=1.6)
    plt.plot(results['DATE_TIME'][:1000], results['Baseline'][:1000], label="Baseline", alpha=0.8)
    plt.plot(results['DATE_TIME'][:1000], results['FFNN'][:1000], label="FFNN", alpha=0.8)
    plt.xlabel("Datetime")
    plt.ylabel(f"{target} (original units)")
    plt.title("Test Set Predictions vs True Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 12) Save
    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        ffnn.save(save_model_path)
        print(f"💾 Model saved to {save_model_path}")

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "metrics": {"Baseline": base_metrics, "FFNN": ffnn_metrics},
        "models": {"Baseline": baseline, "FFNN": ffnn}
    }

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PV Power Forecasting Pipeline")
    parser.add_argument("--weather", type=str, default=DEFAULT_WEATHER_CSV)
    parser.add_argument("--generation", type=str, default=DEFAULT_GEN_CSV)
    parser.add_argument("--irr_thresh", type=float, default=None)
    parser.add_argument("--save_model", type=str, default=None)
    args = parser.parse_args()

    res = run_pipeline(
        weather_csv=args.weather,
        gen_csv=args.generation,
        irradiation_threshold=args.irr_thresh,
        save_model_path=args.save_model,
        ffnn_cfg=dict(hidden_layers=(256,128,64), activation="relu",
                      dropout=0.2, batch_norm=True, l2=1e-5, lr=5e-4),
        fit_cfg=dict(epochs=200, batch_size=64)
    )

    print("\n✅ Final Summary:")
    print("Baseline →", res["metrics"]["Baseline"])
    print("FFNN →", res["metrics"]["FFNN"])