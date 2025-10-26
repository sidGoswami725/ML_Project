import pandas as pd
import numpy as np
import warnings
from math import sqrt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder # Removed OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import time # Added for timing

# ==============================================================================
# --- Model: Gradient Boosting (Ordinal Preprocessing)
# --- Tuning: Using pre-defined params (n_estimators=1200, lr=0.02, etc.)
# --- Preprocessing: Original OrdinalEncoder + Pipeline approach
# ==============================================================================

# Improvement was seen in kaggle score because of this script
# Default GB = 29479.155
# GB-v1 = 22467.133

warnings.filterwarnings("ignore")
np.random.seed(42)

print("Starting Gradient Boosting (Original v3 - Reformatted) script...")

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

# --- PHASE 2: PREPROCESSING SETUP ---

# 2.1 Define Target and Features
TARGET = "HotelValue"
X = train.drop(columns=[TARGET, 'Id']) # Drop Id from features
y = train[TARGET].copy()
y_log = np.log1p(y) # Log transform target

# Keep test set separate for final prediction
X_test_final = test.drop(columns=['Id']) # Drop Id from test features

# 2.2 Define Feature Groups (based on X_train)
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
print(f"Identified {len(num_cols)} numerical and {len(cat_cols)} categorical features.")

# 2.3 Define Preprocessing Pipelines (Original v3 Logic)
numeric_tree = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_tree = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor_tree = ColumnTransformer([
    ("num", numeric_tree, num_cols),
    ("cat", categorical_tree, cat_cols)
], remainder="drop") # Ensure only specified columns are used

# --- PHASE 3: MODEL DEFINITION & CROSS-VALIDATION ---

# 3.1 Define the Full Pipeline (Preprocessor + Specific GB Model)
# Using the exact parameters from the original script
gb_pipe = Pipeline([
    ("pre", preprocessor_tree),
    ("model", GradientBoostingRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        max_features=0.8,
        min_samples_leaf=3,
        random_state=42
    ))
])
print("\nDefined full pipeline with OrdinalEncoder and specific GB parameters.")

# 3.2 Cross-validation (6 folds)
kf = KFold(n_splits=6, shuffle=True, random_state=42)

print("\nRunning cross-validation (6 folds) for evaluation...")
start_time_cv = time.time()
oof_log = cross_val_predict(gb_pipe, X, y_log, cv=kf, method="predict", n_jobs=-1)
end_time_cv = time.time()
print(f"CV finished in {end_time_cv - start_time_cv:.2f} seconds.")

# Calculate and print OOF RMSE on the original scale
oof_orig = np.expm1(oof_log)
# Use y directly here since y_log was derived from it
rmse_orig = sqrt(mean_squared_error(y, oof_orig))
print(f"OOF RMSE (original scale): {rmse_orig:.2f}")
print("-" * 80)

# --- PHASE 4: FINAL MODEL TRAINING & SUBMISSION ---

# 4.1 Train the final model on the ENTIRE training dataset
print("Fitting final GradientBoosting model on all training data...")
start_time_fit = time.time()
gb_pipe.fit(X, y_log)
end_time_fit = time.time()
print(f"Final model training finished in {end_time_fit - start_time_fit:.2f} seconds.")

# 4.2 Make predictions on the preprocessed test data
print("\nMaking predictions on test.csv...")
test_pred_log = gb_pipe.predict(X_test_final) # Use the separate test set
test_pred = np.expm1(test_pred_log)

# 4.3 Create and Save Submission File
submission = pd.DataFrame({
    "Id": test_ids, # Use the stored test IDs
    "HotelValue": test_pred
})
submission.to_csv("submission_gbv1.csv", index=False, float_format="%.6f")

print("\n" + "=" * 80)
print("submission_gbv1.csv has been created!")
print(submission.head())
print("=" * 80)