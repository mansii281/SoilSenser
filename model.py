# model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
import optuna
import joblib

# ========================= CONFIG =========================
DATA_PATH = "/Users/mansisharma/Desktop/Model/soil_dataset.csv"
FEATURES = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
LABELS = ["Healthy", "Low", "Moderate", "High"]

# ========================= LOAD DATA =========================
df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y_reg = df['SDI']
y_cls = df['Level']

le = LabelEncoder()
y_cls_enc = le.fit_transform(y_cls)

X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls_enc, test_size=0.2, random_state=42
)

# ========================= OPTUNA HYPERPARAMETER TUNING =========================
def objective(trial):
    # RandomForest parameters
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
    rf_max_depth = trial.suggest_int("rf_max_depth", 5, 30)

    # XGBoost parameters
    xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
    xgb_max_depth = trial.suggest_int("xgb_max_depth", 2, 12)
    xgb_lr = trial.suggest_float("xgb_lr", 0.01, 0.3)
    xgb_subsample = trial.suggest_float("xgb_subsample", 0.5, 1.0)
    xgb_colsample = trial.suggest_float("xgb_colsample", 0.5, 1.0)

    # Initialize models
    rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, n_jobs=-1, random_state=42)
    xgb = XGBRegressor(
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_lr,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )

    rf.fit(X_train, y_reg_train)
    xgb.fit(X_train, y_reg_train)
    hybrid_preds = (rf.predict(X_test) + xgb.predict(X_test)) / 2
    rmse = mean_squared_error(y_reg_test, hybrid_preds) ** 0.5
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("✅ Best hyperparameters:", best_params)

# ========================= TRAIN HYBRID MODELS =========================
# Regression Hybrid
rf_reg = RandomForestRegressor(n_estimators=best_params["rf_n_estimators"],
                               max_depth=best_params["rf_max_depth"], n_jobs=-1, random_state=42)
xgb_reg = XGBRegressor(
    n_estimators=best_params["xgb_n_estimators"],
    max_depth=best_params["xgb_max_depth"],
    learning_rate=best_params["xgb_lr"],
    subsample=best_params["xgb_subsample"],
    colsample_bytree=best_params["xgb_colsample"],
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

rf_reg.fit(X_train, y_reg_train)
xgb_reg.fit(X_train, y_reg_train)

# Classification Hybrid
rf_cls = RandomForestClassifier(n_estimators=best_params["rf_n_estimators"],
                                max_depth=best_params["rf_max_depth"], n_jobs=-1, random_state=42)
xgb_cls = XGBClassifier(
    n_estimators=best_params["xgb_n_estimators"],
    max_depth=best_params["xgb_max_depth"],
    learning_rate=best_params["xgb_lr"],
    subsample=best_params["xgb_subsample"],
    colsample_bytree=best_params["xgb_colsample"],
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42
)

rf_cls.fit(X_train, y_cls_train)
xgb_cls.fit(X_train, y_cls_train)

# ========================= PREDICTIONS =========================
# Regression
train_reg_preds = (rf_reg.predict(X_train) + xgb_reg.predict(X_train)) / 2
test_reg_preds = (rf_reg.predict(X_test) + xgb_reg.predict(X_test)) / 2

# Classification
train_hybrid_probs = (rf_cls.predict_proba(X_train) + xgb_cls.predict_proba(X_train)) / 2
test_hybrid_probs = (rf_cls.predict_proba(X_test) + xgb_cls.predict_proba(X_test)) / 2
train_cls_preds = le.inverse_transform(np.argmax(train_hybrid_probs, axis=1))
test_cls_preds = le.inverse_transform(np.argmax(test_hybrid_probs, axis=1))
y_cls_train_labels = le.inverse_transform(y_cls_train)
y_cls_test_labels = le.inverse_transform(y_cls_test)

# ========================= METRICS =========================
print("\n📊 REGRESSION METRICS")
print(f"Train RMSE: {mean_squared_error(y_reg_train, train_reg_preds)**0.5:.4f}")
print(f"Test RMSE: {mean_squared_error(y_reg_test, test_reg_preds)**0.5:.4f}")
print(f"Train R²: {r2_score(y_reg_train, train_reg_preds):.4f}")
print(f"Test R²: {r2_score(y_reg_test, test_reg_preds):.4f}")

print("\n📊 CLASSIFICATION METRICS")
print(f"Train Accuracy: {accuracy_score(y_cls_train_labels, train_cls_preds)*100:.2f}%")
print(f"Test Accuracy: {accuracy_score(y_cls_test_labels, test_cls_preds)*100:.2f}%")
print(f"Train F1: {f1_score(y_cls_train_labels, train_cls_preds, average='weighted'):.4f}")
print(f"Test F1: {f1_score(y_cls_test_labels, test_cls_preds, average='weighted'):.4f}")

# Confusion Matrices
train_cm = confusion_matrix(y_cls_train_labels, train_cls_preds, labels=LABELS)
test_cm = confusion_matrix(y_cls_test_labels, test_cls_preds, labels=LABELS)
print("\nTrain Confusion Matrix:")
print(pd.DataFrame(train_cm, index=[f"Actual {l}" for l in LABELS],
                   columns=[f"Pred {l}" for l in LABELS]))
print("\nTest Confusion Matrix:")
print(pd.DataFrame(test_cm, index=[f"Actual {l}" for l in LABELS],
                   columns=[f"Pred {l}" for l in LABELS]))

# ========================= SAVE MODELS =========================
joblib.dump(rf_reg, "rf_reg.pkl")
joblib.dump(xgb_reg, "xgb_reg.pkl")
joblib.dump(rf_cls, "rf_cls.pkl")
joblib.dump(xgb_cls, "xgb_cls.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\n✅ Models saved!")

