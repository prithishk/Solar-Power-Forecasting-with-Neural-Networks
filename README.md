# ☀️ Solar Power Forecasting with Neural Networks

### 🧠 Project Overview

Solar energy forecasting plays a crucial role in optimizing renewable energy utilization, grid balancing, and smart energy management systems.
This project uses **deep learning (Artificial Neural Networks – ANN)** to predict **solar power generation** based on weather parameters like ambient temperature, module temperature, and solar irradiation.

By training and comparing multiple neural network architectures, the goal is to **accurately forecast DC power output** for solar panels, thereby improving plant-level operational efficiency and energy planning.

## 🌍 Motivation

Solar power generation depends heavily on environmental conditions that vary throughout the day.
Accurate short-term forecasting helps:

* ⚡ Optimize energy dispatch and storage.
* 💰 Reduce operational costs for energy providers.
* 🌱 Enhance integration of renewables into smart grids.

## 📂 Dataset Description

The dataset used is from the **Kaggle Solar Power Generation Data**. It includes:

* Weather sensor readings (ambient & module temperature, irradiation).
* Power generation records from multiple inverters across timestamps.

### **1️⃣ Weather Sensor Data**

| Column                | Description                           |
| --------------------- | ------------------------------------- |
| `DATE_TIME`           | Timestamp of measurement              |
| `PLANT_ID`            | Unique plant identifier               |
| `SOURCE_KEY`          | Unique inverter ID                    |
| `AMBIENT_TEMPERATURE` | Ambient air temperature (°C)          |
| `MODULE_TEMPERATURE`  | Solar module surface temperature (°C) |
| `IRRADIATION`         | Solar radiation intensity (kW/m²)     |

### **2️⃣ Power Generation Data**

| Column        | Description                                   |
| ------------- | --------------------------------------------- |
| `DATE_TIME`   | Timestamp of generation record                |
| `PLANT_ID`    | Plant identifier                              |
| `SOURCE_KEY`  | Inverter identifier                           |
| `DC_POWER`    | Direct Current power before inverter (kW)     |
| `AC_POWER`    | Alternating Current power after inverter (kW) |
| `DAILY_YIELD` | Daily cumulative energy (kWh)                 |
| `TOTAL_YIELD` | Lifetime cumulative energy (kWh)              |

## 🧹 Data Preprocessing Pipeline

✅ **Steps performed:**

1. **Data Loading:**
   Imported both CSV files into Pandas DataFrames.

2. **Datetime Parsing:**
   Converted `DATE_TIME` fields into Python datetime objects.

3. **Merging Datasets:**
   Combined weather and generation data based on matching timestamps and `PLANT_ID`.

4. **Missing Value Handling:**
   Checked for and interpolated missing timestamps (25 missing in generation data).

5. **Feature Scaling:**
   Applied `MinMaxScaler` for normalization between 0 and 1.

6. **Train-Test Split:**

   * **Training Set:** 80%
   * **Testing Set:** 20%
   * Shapes:

     * `X_train`: (2555, 3)
     * `y_train`: (2555, 1)
     * `X_test`: (627, 3)
     * `y_test`: (627, 1)

## 🧩 Neural Network Models

Multiple ANN architectures were designed and tested to compare performance on different feature sets.

| Model                                            | Input Features                | Layers         | Regularization      | R² Score   | MAE    | RMSE   |
| ------------------------------------------------ | ----------------------------- | -------------- | ------------------- | ---------- | ------ | ------ |
| **A. Radiation-only (Compact Net)**              | 3 (Irradiation + Temperature) | 3 Dense Layers | None                | **0.9929** | 0.0143 | 0.0260 |
| **B. Radiation + Meteorological (BN + Dropout)** | 3                             | 4 Dense Layers | BatchNorm + Dropout | **0.9926** | 0.0148 | 0.0267 |
| **C. All Features (Deep + L2 Regularized)**      | 3                             | 6 Dense Layers | L2 Regularization   | **0.0287** | 0.2003 | 0.3052 |
| **D. Radiation + Time (Small Net)**              | 3                             | 3 Dense Layers | None                | **0.9925** | 0.0151 | 0.0267 |

## ⚙️ Model Training

### **Framework:** TensorFlow / Keras

* Optimizer: `Adam`
* Loss: `Mean Squared Error (MSE)`
* Metrics: `Mean Absolute Error (MAE)`
* Epochs: 20
* Batch Size: 32

### **Sample Training Log:**

Epoch 1/20
64/64 ━━━━━━━━━━━━ loss: 0.0917 - mae: 0.2443 - val_loss: 0.0693 - val_mae: 0.2096
...
Epoch 20/20
64/64 ━━━━━━━━━━━━ loss: 0.0174 - mae: 0.1003 - val_loss: 0.0133 - val_mae: 0.0856

### **Test Performance:**

MAE (kW): 2771.38
RMSE: 3666.77
R²: 0.8351

## 📈 Model Evaluation and Insights

### ✅ Observations

* **Models A, B, and D** achieved excellent accuracy (R² ≈ 0.99).
* **Model C (deeper net)** overfit heavily, showing poor generalization.
* Irradiation and temperature are **key predictive features**.
* Compact, well-regularized models perform best for this dataset.

### 📊 Key Performance Summary

| Metric               | Training | Validation | Test    |
| -------------------- | -------- | ---------- | ------- |
| **MAE (Normalized)** | 0.0174   | 0.0133     | 0.0951  |
| **RMSE (kW)**        | —        | —          | 3666.77 |
| **R²**               | —        | —          | 0.8351  |

## 🧠 Architecture Visualization

*(Placeholder — you can add a model plot later)*


Input Layer (3 features)
     ↓
Dense(64, relu)
     ↓
Dense(32, relu)
     ↓
Dense(16, relu)
     ↓
Dense(1, linear)

## 💾 Environment Setup

### **Requirements**

| Library      | Version |
| ------------ | ------- |
| Python       | ≥ 3.10  |
| TensorFlow   | ≥ 2.16  |
| Pandas       | ≥ 2.2   |
| NumPy        | ≥ 1.26  |
| scikit-learn | ≥ 1.7   |
| Matplotlib   | ≥ 3.9   |
| Seaborn      | ≥ 0.13  |

### **Setup Instructions**

bash
# 1. Clone the Repository
git clone https://github.com/<your-username>/Solar_Power_Forecasting_with_Neural_Networks.git
cd Solar_Power_Forecasting_with_Neural_Networks

# 2. Create Virtual Environment
python3 -m venv tfenv
source tfenv/bin/activate  # Mac/Linux
# .\tfenv\Scripts\activate  # Windows

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run the Project
python3 pv_power_prediction.py

## 📊 Visual Outputs

**Example visualizations generated by the script:**

1. **Correlation Heatmap** of weather vs power variables.
2. **Training vs Validation Loss Curve.**
3. **Actual vs Predicted Power Output Plot.**
4. **Feature Importance Graph (optional).**

## 🧾 Key Learnings

* The **IRRADIATION** feature is the most influential in power forecasting.
* Neural networks can capture **non-linear relationships** between temperature and DC output.
* Over-regularization or excessive depth can harm model performance.
* Compact architectures with dropout and batch normalization yield robust results.

## 🚀 Future Enhancements

🔹 Incorporate **time-lag features** for sequence prediction.
🔹 Experiment with **LSTM or GRU models** for temporal forecasting.
🔹 Deploy model via **Streamlit or Flask API** for real-time predictions.
🔹 Extend the dataset with **multiple plants and seasonal data**.
🔹 Integrate **weather forecast APIs** to predict future solar generation.

## 🏷️ License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute with appropriate attribution.


## 🌟 Acknowledgements

* Dataset: [Solar Power Generation Data – Kaggle](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
* Frameworks: TensorFlow, scikit-learn
* Environment: macOS, Apple M3 GPU (Metal backend)
