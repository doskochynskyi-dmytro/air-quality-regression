import time
import glob
import os
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")

# 1. Get the absolute path to the directory containing app.py
APP_DIR = os.path.dirname(os.path.abspath(__file__))

weight_path = os.path.join(APP_DIR, "assets", "weights")

print(f"Loading weights from: {weight_path}")

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class AirQualityMLP(nn.Module):
    def __init__(self, input_dim, h1, h2):
        super(AirQualityMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.outputlayer = nn.Linear(h2, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.outputlayer(x)


#describing model types

# Load Scalers
scaler = joblib.load(os.path.join(weight_path, "scaler.joblib"))
poly = joblib.load(os.path.join(weight_path, "poly.joblib"))

# A. HistGradientBoosting (Scikit-Learn)
gradientboost_model = joblib.load(
    os.path.join(weight_path, "gradientboostweights.joblib")
)

# B. PyTorch Polynomial Linear Regression
linear_model = torch.load(
    os.path.join(weight_path, "linearregression.pth"),
    map_location=device,
    weights_only=False  # <-- ADD THIS PARAMETER
)
linear_model.eval()

# C. PyTorch DeepML Model
deepml_model = torch.load(
    os.path.join(weight_path, "deepml.pth"),
    map_location=device,
    weights_only=False  # <-- ADD THIS PARAMETER
)
deepml_model.eval()

scaler_raw = joblib.load(os.path.join(weight_path, "scaler_raw.joblib"))
scaler_poly = joblib.load(os.path.join(weight_path, "scaler_poly.joblib"))
poly = joblib.load(os.path.join(weight_path, "poly.joblib"))

def predict_temperature(
    co_gt, pt08_s1, c6h6_gt, pt08_s2, nox_gt, pt08_s3, no2_gt, pt08_s4, pt08_s5, rh
):
    raw_features = np.array([[
        co_gt, pt08_s1, c6h6_gt, pt08_s2, nox_gt, 
        pt08_s3, no2_gt, pt08_s4, pt08_s5, rh
    ]])

    results = []

    # --- 1. HistGradientBoosting (Raw Features) ---
    t0 = time.perf_counter()
    pred_gb = gradientboost_model.predict(raw_features)[0]
    lat_gb = (time.perf_counter() - t0) * 1000

    results.append({
        "Algorithm": "HistGradientBoosting",
        "Predicted Temperature (°C)": round(float(pred_gb), 2),
        "Inference Latency (ms)": round(lat_gb, 3)
    })

    # --- 2. PyTorch Linear Regression (65 Poly Features -> Scaled) ---
    t0 = time.perf_counter()
    poly_features = poly.transform(raw_features)                     # 10 -> 65 features
    scaled_poly_features = scaler_poly.transform(poly_features)       # Scale 65 features
    tensor_poly = torch.tensor(scaled_poly_features, dtype=torch.float32)

    with torch.no_grad():
        pred_lr = linear_model(tensor_poly).item()
    lat_lr = (time.perf_counter() - t0) * 1000

    results.append({
        "Algorithm": "PyTorch Polynomial Linear Reg",
        "Predicted Temperature (°C)": round(float(pred_lr), 2),
        "Inference Latency (ms)": round(lat_lr, 3)
    })

    # --- 3. PyTorch DeepML MLP (10 Raw Features -> Scaled) ---
    t0 = time.perf_counter()
    scaled_raw_features = scaler_raw.transform(raw_features)         # Scale 10 features
    tensor_scaled = torch.tensor(scaled_raw_features, dtype=torch.float32)

    with torch.no_grad():
        pred_mlp = deepml_model(tensor_scaled).item()
    lat_mlp = (time.perf_counter() - t0) * 1000

    results.append({
        "Algorithm": "PyTorch DeepML MLP",
        "Predicted Temperature (°C)": round(float(pred_mlp), 2),
        "Inference Latency (ms)": round(lat_mlp, 3)
    })

    df_results = pd.DataFrame(results)
    return df_results


with gr.Blocks(title="Air Quality Temperature Predictor") as demo:
    gr.Markdown("# 🌡️ Air Quality Multi-Model Temperature Benchmark")
    gr.Markdown("Enter sensor readings below to run predictions through all 3 algorithms simultaneously.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Sensor Input Readings")
            input_co = gr.Number(label="CO(GT)", value=2.6)
            input_s1 = gr.Number(label="PT08.S1(CO)", value=1360.0)
            input_c6h6 = gr.Number(label="C6H6(GT)", value=11.9)
            input_s2 = gr.Number(label="PT08.S2(NMHC)", value=1046.0)
            input_nox = gr.Number(label="NOx(GT)", value=166.0)
            input_s3 = gr.Number(label="PT08.S3(NOx)", value=1056.0)
            input_no2 = gr.Number(label="NO2(GT)", value=113.0)
            input_s4 = gr.Number(label="PT08.S4(NO2)", value=1692.0)
            input_s5 = gr.Number(label="PT08.S5(O3)", value=1268.0)
            input_rh = gr.Number(label="Relative Humidity RH (%)", value=48.9)

            btn_predict = gr.Button("🚀 Predict Temperature Across All Models", variant="primary")

        with gr.Column():
            gr.Markdown("### Model Performance & Benchmark Results")
            table_output = gr.Dataframe(interactive=False)

    btn_predict.click(
        fn=predict_temperature,
        inputs=[
            input_co, input_s1, input_c6h6, input_s2, input_nox,
            input_s3, input_no2, input_s4, input_s5, input_rh
        ],
        outputs=[table_output]
    )

if __name__ == "__main__":
    demo.launch()
