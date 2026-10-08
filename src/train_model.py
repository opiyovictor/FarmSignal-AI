"""
FarmSignal AI — risk model training.

Trains an XGBoost classifier to predict farmer default/dropout risk, then
uses SHAP to attach a plain-language "top reasons" explanation to every
farmer's score. This is the piece that makes a risk score *actionable* for
a field officer instead of just a number.

Outputs:
  models/risk_model.json      -> trained XGBoost model
  data/risk_scores.csv        -> farmer_id, risk_score, risk_tier, top reasons
  models/global_feature_importance.csv
"""

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
import json
from pathlib import Path

DATA_PATH = "data/farmers.csv"
MODEL_PATH = "models/risk_model.json"
SCORES_PATH = "data/risk_scores.csv"
IMPORTANCE_PATH = "models/global_feature_importance.csv"
METRICS_PATH = "models/model_metrics.json"

CATEGORICAL_COLS = ["region", "gender"]
TARGET_COL = "defaulted_or_dropped_out"
ID_COL = "farmer_id"

# Human-readable direction of each feature, used to turn a SHAP value into a
# one-line reason a non-technical field officer can read.
FEATURE_DESCRIPTIONS = {
    "days_since_last_farm_visit": "long gap since last farm visit",
    "input_redemption_rate": "low input redemption rate",
    "agronomic_training_attendance": "low training attendance",
    "group_repayment_avg": "weak repayment among peer group",
    "prior_season_yield_index": "below-average prior yield",
    "distance_to_agrodealer_km": "long distance to nearest agrodealer",
    "mobile_money_active": "no active mobile money usage",
    "loan_amount_kes": "relatively large loan amount",
    "years_with_program": "still early in program tenure",
    "farm_size_acres": "farm size",
    "age": "age",
}


def load_and_encode():
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    feature_cols = [c for c in encoded.columns if c not in (TARGET_COL, ID_COL)]
    return df, encoded, feature_cols


def risk_tier(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


def top_reasons(shap_row: pd.Series, feature_cols: list[str], n=2) -> str:
    # Only show reasons that push risk UP (positive SHAP contribution).
    contributions = shap_row[feature_cols].sort_values(ascending=False)
    positive = contributions[contributions > 0].head(n)
    reasons = []
    for feat in positive.index:
        base_feat = feat.split("_region_")[0] if "_region_" in feat else feat
        reasons.append(FEATURE_DESCRIPTIONS.get(base_feat, base_feat.replace("_", " ")))
    return "; ".join(reasons) if reasons else "no single dominant driver"


def main():
    df, encoded, feature_cols = load_and_encode()

    X = encoded[feature_cols]
    y = encoded[TARGET_COL]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.90,
        colsample_bytree=0.90,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- Evaluation ---
    test_probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_probs)
    precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
    # Precision at the point where we flag the top ~15% highest-risk farmers
    threshold_15pct = np.quantile(test_probs, 0.85)
    flagged = test_probs >= threshold_15pct
    precision_at_15pct = y_test[flagged].mean() if flagged.sum() else float("nan")

    test_pred = (test_probs >= 0.5).astype(int)
    metrics = {
        "test_roc_auc": round(float(auc), 4),
        "test_accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
        "test_precision": round(float(precision_score(y_test, test_pred, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, test_pred, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, test_pred, zero_division=0)), 4),
        "precision_at_top_15pct": round(float(precision_at_15pct), 4),
        "test_size": int(len(y_test)),
        "random_state": 42,
    }
    Path("models").mkdir(exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test ROC-AUC: {auc:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.1%}")
    print(f"Precision among top 15% highest-risk farmers (threshold={threshold_15pct:.3f}): "
          f"{precision_at_15pct:.1%}")

    model.get_booster().save_model(MODEL_PATH)

    # --- Score every farmer + SHAP explanations ---
    all_probs = model.predict_proba(X)[:, 1]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=X.index)

    reasons = shap_df.apply(lambda row: top_reasons(row, feature_cols), axis=1)

    scores = pd.DataFrame({
        "farmer_id": df[ID_COL],
        "region": df["region"],
        "risk_score": np.round(all_probs, 4),
        "risk_tier": [risk_tier(p) for p in all_probs],
        "top_reasons": reasons,
        "in_test_set": df.index.isin(idx_test),
    })
    scores.to_csv(SCORES_PATH, index=False, encoding="utf-8")
    print(f"Wrote per-farmer risk scores -> {SCORES_PATH}")
    print(scores["risk_tier"].value_counts())

    # --- Global feature importance (for the monitoring dashboard) ---
    global_importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    global_importance.to_csv(IMPORTANCE_PATH, index=False, encoding="utf-8")
    print(f"Wrote global feature importance -> {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
