import pandas as pd  # data handling and analysis
import numpy as np   # numerical computing
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import math
import random

# Load the weather sensor data and generation data
weather_df = pd.read_csv('/Users/admin/Desktop/GIT/Prithish/Solar_Power_Forecasting_with_Neural_Networks/Plant_1_Weather_Sensor_Data.csv')
generation_df = pd.read_csv('/Users/admin/Desktop/GIT/Prithish/Solar_Power_Forecasting_with_Neural_Networks/Plant_1_Generation_Data.csv')

print("Weather Data Sample:")
display(weather_df.head())

print("Generation Data Sample:")
display(generation_df.head())

# Check the last 5 rows
generation_df.tail()
weather_df.tail()

# Check number of inverters per time
print(
    'The number of inverter for data_time {} is {}'.format(
        '15-05-2020 23:00', generation_df[generation_df.DATE_TIME == '15-05-2020 23:00']['SOURCE_KEY'].nunique())
)

# Convert timestamps to datetime 
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'],dayfirst=True)
generation_df['DATE_TIME'] = pd.to_datetime(generation_df['DATE_TIME'],dayfirst=True)

# Find timestamps in weather missing from generation
missing_in_generation = weather_df[~weather_df['DATE_TIME'].isin(generation_df['DATE_TIME'])]
print(f"Missing timestamps in generation data: {len(missing_in_generation)}")

# Step 1: Group by timestamp and sum across inverters
generation_grouped = generation_df.groupby('DATE_TIME')[['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']].sum()

# Step 2: Reindex to match the weather timestamps exactly
generation_grouped = generation_grouped.reindex(weather_df['DATE_TIME']).fillna(0)

# Step 3: Reset index to make DATE_TIME a column again
generation_grouped = generation_grouped.reset_index().rename(columns={'index': 'DATE_TIME'})

# Describe
weather_df_idx = weather_df.set_index('DATE_TIME')
generation_idx = generation_grouped.set_index('DATE_TIME')
display(weather_df_idx.describe())
display(generation_idx.describe())

# Check if both indexes are equal
print("Indexes are equal:" , weather_df_idx.index.equals(generation_idx.index))

# Check for NaNs in weather and generation data
print("Missing values in weather_df:")
print(weather_df_idx.isna().sum())

print("\nMissing values in generation_df:")
print(generation_idx.isna().sum())

# Drop unused columns
weather_df_idx = weather_df_idx.drop(['PLANT_ID', 'SOURCE_KEY'], axis=1)
generation_idx = generation_idx.drop(['DC_POWER', 'TOTAL_YIELD'], axis=1)

# Final Merge
df = pd.merge(generation_idx, weather_df_idx, on='DATE_TIME')
df = df.reset_index()
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
df['DATE'] = df['DATE_TIME'].dt.date

# Split train/test by date
unique_dates = df['DATE'].drop_duplicates().values
np.random.seed(42)
np.random.shuffle(unique_dates)

split_ratio = 0.8
split_index = int(len(unique_dates) * split_ratio)
train_dates = unique_dates[:split_index]
test_dates  = unique_dates[split_index:]

train_df = df[df['DATE'].isin(train_dates)]
test_df  = df[df['DATE'].isin(test_dates)]

# Select features and target
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
target = 'AC_POWER'

# Scaling
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_train = X_scaler.fit_transform(train_df[features])
y_train = y_scaler.fit_transform(train_df[[target]])
X_test = X_scaler.transform(test_df[features])
y_test = y_scaler.transform(test_df[[target]])

print('The X_trian shape is:', X_train.shape)
print('The y_train shape is:', y_train.shape)
print('The X_test shape is:', X_test.shape)
print('The y_test shape is:', y_test.shape)

# TF/Keras Model (simple linear regression net)
activation_function = 'linear'
model = keras.Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(1, activation=activation_function))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=1)

loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.4f}")

# Inverse transform for original scale comparison
y_pred_scaled = model.predict(X_test).flatten()
y_true_scaled = y_test.flatten()
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
results = pd.DataFrame({'true': y_true, 'pred': y_pred})
results['DATE_TIME'] = test_df['DATE_TIME'].values
results['DATE'] = pd.to_datetime(results['DATE_TIME']).dt.date

print(f"MAE (kw): {mean_absolute_error(y_true, y_pred):.4f}")
import numpy as np
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2_score(y_true, y_pred):.4f}")

plt.figure(figsize=(12, 5))
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Model Predictions vs True AC POWER Kw')
plt.xlabel('Time Step')
plt.ylabel('AC_POWER (kW)')
plt.show()

# Advanced: Define, train, and evaluate a deeper FFNN with flexible feature selections and configs
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def to_dataframe(X):
    """Ensure X is a DataFrame with column names f0.. or preserved names."""
    if hasattr(X, "columns"):
        return X
    X = np.asarray(X)
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols)

def detect_or_create_val(Xtr, ytr, Xval=None, yval=None, val_size=0.1, seed=42):
    if Xval is not None and yval is not None:
        return Xtr, ytr, Xval, yval
    Xtr2, Xval2, ytr2, yval2 = train_test_split(Xtr, ytr, test_size=val_size, random_state=seed, shuffle=True)
    return Xtr2, ytr2, Xval2, yval2

def inverse_if_scaled(y, preds):
    if not Y_IS_SCALED:
        return y, preds
    if TARGET_SCALER_NAME in globals():
        scaler = globals()[TARGET_SCALER_NAME]
        y_true = scaler.inverse_transform(np.asarray(y).reshape(-1,1)).ravel()
        y_pred = scaler.inverse_transform(np.asarray(preds).reshape(-1,1)).ravel()
        return y_true, y_pred
    print(f"[WARN] Y_IS_SCALED=True but '{TARGET_SCALER_NAME}' not found. Using scaled values for metrics.")
    return y, preds

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def build_ffnn(input_dim, hidden_layers=(64,64), activation="relu", dropout=0.0, batch_norm=False, l2=0.0, lr=1e-3):
    model = keras.Sequential()
    for i, units in enumerate(hidden_layers):
        if i == 0:
            model.add(keras.layers.Dense(units, activation=activation,
                                         kernel_regularizer=keras.regularizers.l2(l2) if l2>0 else None,
                                         input_shape=(input_dim,)))
        else:
            model.add(keras.layers.Dense(units, activation=activation,
                                         kernel_regularizer=keras.regularizers.l2(l2) if l2>0 else None))
        if batch_norm:
            model.add(keras.layers.BatchNormalization())
        if dropout and dropout>0:
            model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation="linear"))
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model

def train_and_eval(name, Xtr, ytr, Xval, yval, Xte, yte, model_kwargs, fit_kwargs):
    set_seeds(42)
    model = build_ffnn(Xtr.shape[1], **model_kwargs)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=7, factor=0.5, min_lr=1e-6)
    ]
    history = model.fit(Xtr, ytr, validation_data=(Xval, yval),
                        callbacks=callbacks, verbose=0, **fit_kwargs)
    preds = model.predict(Xte, verbose=0).ravel()
    y_true, y_pred = inverse_if_scaled(yte, preds)
    m = metrics(y_true, y_pred)
    return {"name": name, "model": model, "history": history, "preds": y_pred, "metrics": m}

def subset_by_names(Xdf, wanted_names):
    cols_lower = {c.lower(): c for c in Xdf.columns}
    selected = []
    for wn in wanted_names:
        key = wn.lower()
        if key in cols_lower:
            selected.append(cols_lower[key])
    return Xdf[selected] if selected else Xdf

Y_IS_SCALED = False
TARGET_SCALER_NAME = "y_scaler"
CAND_RADIATION = ["GHI", "DNI", "DHI", "irradiance", "SolarIrradiance", "SolarRad"]
CAND_METEO = ["Temperature", "Temp", "Tair", "Humidity", "RH", "WindSpeed", "Wind", "CloudCover", "Clouds"]
CAND_TIME = ["Hour", "hour", "DayOfYear", "doy", "Month", "WeekOfYear", "minute_sin", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]

assert "X_train" in globals() and "y_train" in globals() and "X_test" in globals() and "y_test" in globals(), \
    "Please run the earlier notebook cells to define X_train, y_train, X_test, y_test."
Xtr_df = to_dataframe(globals()["X_train"])
Xte_df = to_dataframe(globals()["X_test"])
ytr = np.asarray(globals()["y_train"]).ravel()
yte = np.asarray(globals()["y_test"]).ravel()
Xval_raw = globals().get("X_val", None)
yval_raw = globals().get("y_val", None)
Xtr_df, ytr, Xval_df, yval = detect_or_create_val(Xtr_df, ytr,
                                                  to_dataframe(Xval_raw) if Xval_raw is not None else None,
                                                  np.asarray(yval_raw).ravel() if yval_raw is not None else None)

all_feature_names = list(Xtr_df.columns)
FS_A = list(dict.fromkeys(subset_by_names(Xtr_df, CAND_RADIATION).columns))
FS_B = list(dict.fromkeys(subset_by_names(Xtr_df, CAND_RADIATION + CAND_METEO).columns))
FS_C = all_feature_names
FS_D = list(dict.fromkeys(subset_by_names(Xtr_df, CAND_RADIATION + CAND_TIME).columns))
feature_sets = {
    "A: Radiation-only (compact net)": FS_A,
    "B: Rad+Meteo (BN+Dropout)": FS_B,
    "C: ALL features (deeper+L2)": FS_C,
    "D: Rad+Time (small net)": FS_D,
}
model_cfgs = {
    "A: Radiation-only (compact net)": dict(hidden_layers=(64,64), activation="relu", dropout=0.0, batch_norm=False, l2=0.0, lr=1e-3),
    "B: Rad+Meteo (BN+Dropout)"     : dict(hidden_layers=(128,128,64), activation="relu", dropout=0.2, batch_norm=True, l2=0.0, lr=5e-4),
    "C: ALL features (deeper+L2)"   : dict(hidden_layers=(256,256,128,64), activation="relu", dropout=0.3, batch_norm=True, l2=1e-4, lr=1e-4),
    "D: Rad+Time (small net)"       : dict(hidden_layers=(32,32), activation="relu", dropout=0.0, batch_norm=False, l2=0.0, lr=1e-3),
}
fit_cfgs = {
    "epochs": 200,
    "batch_size": 256,
    "shuffle": True,
}
results = []
for name, cols in feature_sets.items():
    Xtr_sel = Xtr_df[cols].values
    Xval_sel = Xval_df[cols].values
    Xte_sel  = Xte_df[cols].values
    res = train_and_eval(name, Xtr_sel, ytr, Xval_sel, yval, Xte_sel, yte,
                         model_kwargs=model_cfgs[name], fit_kwargs=fit_cfgs)
    results.append(res)
    print(f"{name}: {res['metrics']} (features used: {len(cols)})")

metrics_df = pd.DataFrame(
    {r["name"]: r["metrics"] for r in results}
).T[["MAE","RMSE","R2"]].sort_values("RMSE")
display(metrics_df.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","R2":"{:.3f}"}))

PLOT_MAX = 1500
idx = np.arange(min(len(yte), PLOT_MAX))
y_plot_true, _ = inverse_if_scaled(yte[idx], yte[idx])

plt.figure(figsize=(12,6))
plt.plot(idx, y_plot_true, label="Ground truth", linewidth=2)
for r in results:
    plt.plot(idx, r["preds"][idx], label=r["name"], alpha=0.9)
plt.title("PV generation: test set predictions (4 models)")
plt.xlabel("Sample index (test subset)")
plt.ylabel("Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
model.save("solar_forecast_best_model.h5")