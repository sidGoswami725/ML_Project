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
import time # Added for timing

# ==============================================================================
# --- Model: Gradient Boosting (Micro-Ensemble + Feature Selection)
# --- Tuning: Using pre-defined params
# --- Preprocessing: Original OrdinalEncoder + Feature Selection approach
# ==============================================================================

# Score got kinda worse
# Default GB = 29479.155
# GB-v1 = 22467.133
# GB-v2 = 22470.431

warnings.filterwarnings("ignore")
np.random.seed(42)

print("Starting Gradient Boosting (v6 Micro-Ensemble Reformatted) script...")

# --- PHASE 1: LOAD DATA ---
try:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Error: train.csv or test.csv not found.")
    exit()

print(f"Original train shape: {train.shape}")
print(f"Original test shape: {test.shape}")
test_ids = test['Id'] # Store test IDs

# --- PHASE 2: INITIAL PREPROCESSING SETUP ---

# 2.1 Define Target and Initial Features
TARGET = "HotelValue"
X = train.drop(columns=[TARGET, 'Id']) # Drop Id from features
y = train[TARGET].copy()
y_log = np.log1p(y) # Log transform target

# Keep test set separate
X_test_initial = test.drop(columns=['Id']) # Drop Id from test features

# 2.2 Define Initial Feature Groups
num_cols_initial = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_initial = X.select_dtypes(include=['object','category']).columns.tolist()
print(f"Initial feature set: {len(num_cols_initial)} numerical, {len(cat_cols_initial)} categorical.")

# 2.3 Define Initial Preprocessing Pipelines (OrdinalEncoder)
numeric_pipe_initial = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe_initial = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor_initial = ColumnTransformer([
    ("num", numeric_pipe_initial, num_cols_initial),
    ("cat", categorical_pipe_initial, cat_cols_initial)
], remainder="drop")

# --- PHASE 3: FEATURE SELECTION ---

# 3.1 Define and Train Base Model for Feature Importance
# Using parameters from original script for the base model
base_model_pipeline = Pipeline([
    ("pre", preprocessor_initial),
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

print("\n--- Running Feature Selection ---")
print("Step 1: Training base GBM (seed 42) to compute feature importances...")
start_time_fi = time.time()
base_model_pipeline.fit(X, y_log)
end_time_fi = time.time()
print(f"Base model training finished in {end_time_fi - start_time_fi:.2f} seconds.")

# 3.2 Extract and Rank Feature Importances
try:
    # Get feature names after preprocessing
    feature_names = base_model_pipeline.named_steps["pre"].get_feature_names_out()
    importances = base_model_pipeline.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
except Exception as e:
    print(f"Warning: Could not automatically get feature names. Using original column names. Error: {e}")
    # Fallback if get_feature_names_out fails
    all_cols = num_cols_initial + cat_cols_initial
    if len(all_cols) == len(base_model_pipeline.named_steps["model"].feature_importances_):
         importances = base_model_pipeline.named_steps["model"].feature_importances_
         fi = pd.DataFrame({"feature": all_cols, "importance": importances})
         fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        print("Error: Mismatch in feature count after preprocessing. Skipping feature selection.")
        fi = pd.DataFrame() # Ensure fi exists but is empty


# 3.3 Select Top Features (Only if feature importances were successfully extracted)
if not fi.empty:
    keep_ratio = 0.6
    n_keep = int(len(fi) * keep_ratio)
    keep_features_transformed = fi.iloc[:n_keep]["feature"].tolist()
    print(f"Identified top {n_keep}/{len(fi)} transformed features ({keep_ratio*100:.0f}%).")

    # 3.4 Map Transformed Feature Names Back to Original Columns (Best effort)
    # This part is tricky as OrdinalEncoder doesn't create new names like OneHotEncoder
    # We'll keep the original columns that were part of the top transformed features
    # Note: This might keep slightly more columns than strictly necessary but is safer
    cols_to_keep = []
    for orig_col in X.columns:
        # Check if the original column name (or a potential transformed name derived from it)
        # is present in the list of top transformed features.
        # This checks num_ features (e.g., 'num__colname') and cat_ features (e.g., 'cat__colname')
        if f'num__{orig_col}' in keep_features_transformed or \
           f'cat__{orig_col}' in keep_features_transformed or \
           orig_col in keep_features_transformed: # Fallback if names weren't prefixed
            cols_to_keep.append(orig_col)

    if not cols_to_keep: # Safety check
         print("Warning: Could not map features. Keeping all original columns.")
         cols_to_keep = X.columns.tolist()

    print(f"Step 2: Selecting corresponding original columns. Keeping {len(cols_to_keep)} columns.")
else:
    # If feature importance failed, keep all original columns
    print("Skipping feature selection step due to issues in importance calculation.")
    cols_to_keep = X.columns.tolist()


# 3.5 Create Filtered Datasets
X_sel = X[cols_to_keep].copy()
X_test_sel = X_test_initial[cols_to_keep].copy()

# 3.6 Update Column Groups for the Selected Features
num_cols_sel = X_sel.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_sel = X_sel.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Selected feature set: {len(num_cols_sel)} numerical, {len(cat_cols_sel)} categorical.")
print("-" * 80)


# --- PHASE 4: MICRO-ENSEMBLE TRAINING & PREDICTION ---
print("\n--- Running Micro-Ensemble Training ---")
seeds = [42, 101, 202, 303, 404]
preds_all = [] # To store predictions from each model in the ensemble
oof_scores = [] # To store OOF scores from each model

kf = KFold(n_splits=6, shuffle=True, random_state=42) # Using 6 folds as in original

for seed in seeds:
    print(f"\n===== Training Ensemble Member (seed {seed}) =====")
    np.random.seed(seed) # Set seed for this member

    # Define the preprocessor *specifically for the selected columns*
    preprocessor_sel = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols_sel),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), cat_cols_sel)
    ], remainder="drop")

    # Define the model pipeline for this ensemble member
    # Using parameters from original script's ensemble loop
    gb_member_pipe = Pipeline([
        ("pre", preprocessor_sel),
        ("model", GradientBoostingRegressor(
            n_estimators=1800, # Note: n_estimators is higher here as per original script
            learning_rate=0.012,
            max_depth=5,
            subsample=0.8,
            max_features=0.45,
            min_samples_split=18,
            min_samples_leaf=2,
            random_state=seed # Use the specific seed
        ))
    ])

    # Calculate OOF predictions and score for this member
    print(f"Running 6-fold CV for seed {seed}...")
    start_time_cv_seed = time.time()
    oof_log_seed = cross_val_predict(gb_member_pipe, X_sel, y_log, cv=kf, method="predict", n_jobs=-1)
    end_time_cv_seed = time.time()
    print(f"CV for seed {seed} finished in {end_time_cv_seed - start_time_cv_seed:.2f} seconds.")

    oof_orig_seed = np.expm1(oof_log_seed)
    rmse_seed = sqrt(mean_squared_error(y, oof_orig_seed)) # Compare OOF against original y
    oof_scores.append(rmse_seed)
    print(f"OOF RMSE (original scale, seed {seed}): {rmse_seed:.2f}")

    # Train this member on the full selected training data
    print(f"Fitting final model for seed {seed} on selected features...")
    start_time_fit_seed = time.time()
    gb_member_pipe.fit(X_sel, y_log)
    end_time_fit_seed = time.time()
    print(f"Training for seed {seed} finished in {end_time_fit_seed - start_time_fit_seed:.2f} seconds.")

    # Generate predictions on the selected test data
    test_pred_log_seed = gb_member_pipe.predict(X_test_sel)
    test_pred_seed = np.expm1(test_pred_log_seed)
    preds_all.append(test_pred_seed)

print("-" * 80)
print(f"\nAverage OOF RMSE across {len(seeds)} seeds: {np.mean(oof_scores):.2f}")
print("-" * 80)

# --- PHASE 5: FINAL BLENDING & SUBMISSION ---

# 5.1 Average the predictions from all ensemble members
print("\nAveraging predictions from all ensemble members...")
final_pred = np.mean(preds_all, axis=0)

# 5.2 Create and Save Submission File
submission = pd.DataFrame({
    "Id": test_ids,
    "HotelValue": final_pred
})
submission.to_csv("submission_gbv2.csv", index=False, float_format="%.6f")

print("\n" + "=" * 80)
print("submission_gbv2.csv has been created!")
print(submission.head())
print("=" * 80)