# =====================================================
# GradBoost_v3.py
# GradientBoosting Micro-Ensemble (5 Seeds)
# + Seed Jitter + 8-Fold CV + Target Encoding
# =====================================================

# Best score yet
# Score got kinda worse
# Default GB = 29479.155
# GB-v1 = 22467.133
# GB-v2 = 22470.431
# GB-v3 = 21554.736

import pandas as pd
import numpy as np
import warnings
from math import sqrt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============== Load data ==============
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

TARGET = "HotelValue"
X = train.drop(columns=[TARGET])
y = train[TARGET].copy()
y_log = np.log1p(y)

print(f"Train shape: {X.shape}, Test shape: {test.shape}")

# ============== Identify feature types ==============
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# ============== Target Encoding (Top 3 categorical features) ==============
# Select top categorical features by cardinality (or tweak manually)
top_cats = sorted(cat_cols, key=lambda c: X[c].nunique(), reverse=True)[:3]
print(f"Target-encoding categorical features: {top_cats}")

for col in top_cats:
    means = train.groupby(col)[TARGET].mean()
    X[f"{col}_te"] = X[col].map(means)
    test[f"{col}_te"] = test[col].map(means)
    # handle unseen categories
    test[f"{col}_te"].fillna(y.mean(), inplace=True)
    num_cols.append(f"{col}_te")

# ============== Preprocessing ==============
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
], remainder="drop")

# ============== Step 1: Base GBM for Feature Importance ==============
base_model = Pipeline([
    ("pre", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=1500,
        learning_rate=0.012,
        max_depth=5,
        subsample=0.8,
        max_features=0.45,
        min_samples_split=18,
        min_samples_leaf=2,
        random_state=42
    ))
])

print("\nComputing feature importances using seed 42 GBM...")
base_model.fit(X, y_log)

# Get transformed feature names
if hasattr(base_model.named_steps["pre"], "get_feature_names_out"):
    feature_names = base_model.named_steps["pre"].get_feature_names_out()
else:
    feature_names = num_cols + cat_cols

importances = base_model.named_steps["model"].feature_importances_
fi = pd.DataFrame({"feature": feature_names, "importance": importances})
fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)

# Keep top 60% most important features
keep_ratio = 0.6
n_keep = int(len(fi) * keep_ratio)
keep_features = fi.iloc[:n_keep]["feature"].tolist()
print(f"Keeping top {n_keep}/{len(fi)} features ({keep_ratio*100:.0f}%).")

# Filter columns
cols_to_keep = [col for col in X.columns if any(col in f for f in keep_features)]
if not cols_to_keep:
    cols_to_keep = X.columns.tolist()

X_sel = X[cols_to_keep]
test_sel = test[cols_to_keep]

print(f"Reduced feature set: {len(cols_to_keep)} columns kept.")

# ============== Update column groups ==============
num_cols = X_sel.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_sel.select_dtypes(include=['object', 'category']).columns.tolist()

# ============== Micro-Ensemble with Seed Jitter ==============
seeds = [42, 101, 202, 303, 404]
preds_all = []
kf = KFold(n_splits=8, shuffle=True, random_state=42)

for seed in seeds:
    # introduce small stochastic variation per seed
    n_estimators = int(1700 + np.random.randint(-150, 150))
    learning_rate = 0.0115 + 0.0005 * np.random.rand()
    subsample = np.clip(0.75 + 0.1 * np.random.rand(), 0.7, 0.9)
    max_features = np.clip(0.4 + 0.1 * np.random.rand(), 0.35, 0.55)

    print(f"\n===== Training with seed {seed} =====")
    print(f"(n_estimators={n_estimators}, lr={learning_rate:.4f}, subsample={subsample:.2f}, max_features={max_features:.2f})")
    np.random.seed(seed)

    gb_model = Pipeline([
        ("pre", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]), cat_cols)
        ], remainder="drop")),
        ("model", GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            subsample=subsample,
            max_features=max_features,
            min_samples_split=18,
            min_samples_leaf=2,
            random_state=seed
        ))
    ])

    # OOF validation
    oof_log = cross_val_predict(gb_model, X_sel, y_log, cv=kf, method="predict", n_jobs=-1)
    oof_orig = np.expm1(oof_log)
    rmse_seed = sqrt(mean_squared_error(np.expm1(y_log), oof_orig))
    print(f"OOF RMSE (orig scale, seed {seed}): {rmse_seed:.2f}")

    # Fit final model on full training data
    gb_model.fit(X_sel, y_log)
    test_pred_log = gb_model.predict(test_sel)
    test_pred = np.expm1(test_pred_log)
    preds_all.append(test_pred)

# ============== Final Blend ==============
final_pred = np.mean(preds_all, axis=0)

# ============== Save Submission ==============
if "Id" not in test.columns:
    test["Id"] = range(1, len(test) + 1)

submission = pd.DataFrame({
    "Id": test["Id"],
    "HotelValue": final_pred
})
submission.to_csv("submission_gbv3.csv", index=False, float_format="%.6f")

print("\nSaved submission_gbv3.csv")
print(submission.head())


