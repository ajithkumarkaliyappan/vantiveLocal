
# Install sagemaker (if not already). In notebooks, you can run: !pip install sagemaker
# For scripts, ensure sagemaker is installed in your environment.

import os
import csv
import json
import time
import boto3
import numpy as np
from sagemaker.local import LocalSession
from sagemaker.model import Model
from sagemaker.image_uris import retrieve

# -------------------------
# CONFIG — update these
# -------------------------
region        = "us-east-1"   # Change to your AWS region
role_arn      = "arn:aws:iam::365628041686:role/HCL-User-Role-SagemakerAccess-Vantive_sagemaker"   # not used by local, but required by API
model_art_uri = "s3://patientsummarizewebserver/vantive-test-regression-01-13-copy-02/output/model.tar.gz"  # OR "file://<local-path-to-model.tar.gz>"
test_csv_path = r"./insurance_test.csv"        # local path to the test CSV (label first, no header)

# If your test CSV is in S3 instead of local, download it first:
# import boto3
# s3 = boto3.client('s3', region_name=region)
# s3.download_file('<bucket>', '<prefix>/insurance_test.csv', test_csv_path)

# -------------------------
# Local SageMaker Session
# -------------------------
# NOTE: Local Mode requires Docker daemon running.
local_session = LocalSession()
local_session.config = {'local': {'local_code': True}}
boto_session = boto3.Session(region_name=region)

# Resolve the official Linear Learner inference image for your region
image_uri = retrieve(framework="linear-learner", region=region)

# -------------------------
# Create the Model object
# -------------------------
model_name = f"ll-insurance-local-{int(time.time())}"
ll_model = Model(
    model_data=model_art_uri,       # points to your model.tar.gz; can be s3:// or file://
    image_uri=image_uri,            # Linear Learner inference image
    role=role_arn,                  # still required by SDK
    sagemaker_session=local_session,
    name=model_name,
)

# -------------------------
# Deploy locally
# -------------------------
# Local Mode instance_type MUST be "local"
predictor = ll_model.deploy(
    initial_instance_count=1,
    instance_type="local",
    endpoint_name=model_name
)

# Configure content type and expected output
predictor.content_type = "text/csv"
predictor.accept = "application/jsonlines"

# -------------------------
# Helper: read test CSV
# -------------------------
def read_test_csv_labels_and_features(csv_path):
    """
    Returns:
        y_true (list[float]): labels from first column
        feature_lines (list[str]): CSV lines with features only (no label), to send to predictor
    """
    y_true = []
    feature_lines = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            # skip empty lines
            if not row or all(str(x).strip() == "" for x in row):
                continue
            y_true.append(float(row[0]))
            # features only (remove label)
            features = row[1:]
            # Ensure consistent number of features per row (your preprocess script already does this)
            feature_lines.append(",".join(features))
    return y_true, feature_lines

# -------------------------
# Predict in batches
# -------------------------
def predict_jsonlines(predictor, feature_lines, batch_size=32):
    """
    Send features to local endpoint (Linear Learner) and return predictions (list[float]).
    Linear Learner returns JSONLines: {"predictions":[{"score": <float>}]}
    """
    preds = []
    # Local Mode predictor expects one line per request for CSV payloads.
    # To stay simple and aligned with output order, send one line at a time.
    for line in feature_lines:
        resp = predictor.predict(line)  # returns a bytes/str JSON line
        # The SDK returns a byte-string; decode & parse
        if isinstance(resp, bytes):
            resp = resp.decode("utf-8")
        obj = json.loads(resp)
        score = float(obj["predictions"][0]["score"])
        preds.append(score)
    return preds

# -------------------------
# Metrics
# -------------------------
def evaluate_regression(y_true, y_pred):
    assert len(y_true) == len(y_pred), f"Length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}"
    n = len(y_true)
    mae  = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    mse  = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
    rmse = mse ** 0.5
    # R^2 (coefficient of determination)
    y_mean = sum(y_true) / n
    ss_tot = sum((t - y_mean) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# -------------------------
# Run evaluation
# -------------------------
y_true, feature_lines = read_test_csv_labels_and_features(test_csv_path)

print(f"Loaded test rows: {len(y_true)}")
assert len(y_true) > 0, "Test dataset appears empty."

y_pred = predict_jsonlines(predictor, feature_lines, batch_size=32)

metrics = evaluate_regression(y_true, y_pred)
print("=== Evaluation Metrics (Test) ===")
for k, v in metrics.items():
    print(f"{k}: {v:.6f}")

# -------------------------
# Cleanup local endpoint
# -------------------------
try:
    predictor.delete_endpoint(delete_endpoint_config=True)
    print(f"Deleted local endpoint: {model_name}")
except Exception as e:
    print(f"Cleanup warning: {e}")
