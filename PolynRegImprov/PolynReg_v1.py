import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings

# ==============================================================================
# --- Model: Polynomial Regression (degree=2) + Ridge Regularization
# --- Tuning: Ridge Alpha
# --- Preprocessing: User-provided (modified for Pipeline)
# ==============================================================================

# The score got worse from 26210.063 to 27859.841
# Let's try testing from alpha=1 to 100 in the next version and see if score improves.

warnings.filterwarnings("ignore")
print("Starting Polynomial + Ridge tuning script...")

# --- 1. Load Data ---
try:
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Error: train.csv or test.csv not found.")
    exit()

print(f"Original train shape: {df_train.shape}")
print(f"Original test shape: {df_test.shape}")
test_ids = df_test['Id']

# --- PHASE 2: PREPROCESSING (Pipeline-based approach) ---
# We must use a pipeline here to combine PolyFeatures and Ridge

# 2.1 Log Transform Target Variable
df_train['HotelValue'] = np.log1p(df_train['HotelValue'])
y_train = df_train.pop('HotelValue')
X_train = df_train.drop('Id', axis=1)
X_test_final = df_test.drop('Id', axis=1)

# 2.2 Feature Engineering
for df in [X_train, X_test_final]:
    df['HotelAge'] = df['YearSold'] - df['ConstructionYear']
    df['YearsSinceRenovation'] = df['YearSold'] - df['RenovationYear']
    df['TotalSF'] = df['BasementTotalSF'] + df['GroundFloorArea'] + df['UpperFloorArea']
    df['TotalBathrooms'] = (df['FullBaths'] + (0.5 * df['HalfBaths']) +
                              df['BasementFullBaths'] + (0.5 * df['BasementHalfBaths']))

# 2.3 Define Numerical and Categorical Columns (after engineering)
numerical_cols = X_train.select_dtypes(include=np.number).columns
categorical_cols = X_train.select_dtypes(include='object').columns

# 2.4 Create pipelines
# Numerical pipeline: Impute, Create Poly Features, Scale
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

# Categorical pipeline: Impute, Fill 'None', One-Hot Encode
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Create the full preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ],
    remainder='passthrough'
)

# --- PHASE 3: MODEL PIPELINE & TUNING ---

# 3.1 Create the full model pipeline
# This pipeline does: Preprocessing -> PolyFeatures -> Ridge
poly_ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Ridge(max_iter=3000))
])

# 3.2 Hyperparameter Tuning
# We are tuning the 'alpha' of the 'model' (Ridge)
param_grid = {
    "model__alpha": [10, 25, 50, 100, 200]
}

print("Tuning Polynomial+Ridge (searching for best alpha)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    poly_ridge_pipeline,
    param_grid,
    cv=kf,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print(f"Best RMSLE (CV Score): {-grid_search.best_score_:.6f}")
print(f"Best Params: {grid_search.best_params_}")
print("-" * 80)

# 3.3 Train Final Model
print("Training final model on all data...")
final_model = grid_search.best_estimator_
final_model.fit(X_train, y_train)

# 3.4 Generate Submission
print("Making predictions on test.csv...")
log_predictions = final_model.predict(X_test_final)
final_predictions = np.expm1(log_predictions)

submission_df = pd.DataFrame({"Id": test_ids, "HotelValue": final_predictions})
submission_df.to_csv("submission_polyv1.csv", index=False)

print("\n" + "=" * 80)
print("submission_polyv1.csv has been created!")
print(submission_df.head())
print("=" * 80)