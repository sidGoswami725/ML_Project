import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
from sklearn.exceptions import ConvergenceWarning

# --- Import all required models ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)

# --- 1. Load Data ---
try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    print("Error: train.csv not found.")
    print("Please make sure the file is in the same directory as the script.")
    exit()

print("Data loaded successfully.")

# ==============================================================================
# --- EXPERIMENT 2: NEW STEPS ---
# ==============================================================================

# --- 2. Apply Log Transform to Target Variable (FIXES SKEW) ---
# We use np.log1p which is log(1 + x) to avoid errors if price is 0
y_log = np.log1p(df["HotelValue"])
print("Target variable: 'HotelValue' has been log-transformed (np.log1p).")

# --- 3. Feature Engineering ---
# Create a copy to avoid changing the original dataframe
X = df.drop(["HotelValue", "Id"], axis=1).copy()

# Create new features based on our EDA
X["TotalBathrooms"] = (
    X["FullBaths"]
    + 0.5 * X["HalfBaths"]
    + X["BasementFullBaths"]
    + 0.5 * X["BasementHalfBaths"]
)
X["HotelAge"] = X["YearSold"] - X["ConstructionYear"]
X["IsRemodeled"] = (X["ConstructionYear"] != X["RenovationYear"]).astype(int)
X["TotalSqFt"] = X["GroundFloorArea"] + X["UpperFloorArea"] + X["BasementTotalSF"]

print("New features created: TotalBathrooms, HotelAge, IsRemodeled, TotalSqFt")
# ==============================================================================

# --- 4. Clean Up Columns ---
cols_to_drop = ["PoolQuality", "ExtraFacility", "ServiceLaneType", "BoundaryFence"]
X = X.drop(columns=cols_to_drop)

# Identify numerical and categorical features
# Our new features are all numerical
numerical_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(include="object").columns

print(f"Running evaluation on {X.shape[1]} features.")
print("Metric: RMSLE (Root Mean Squared Logarithmic Error). Lower is better.\n")
print("=" * 80 + "\n")

# --- 5. Create Standard Preprocessing Pipelines ---
# (These are the same as before)

num_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols),
    ],
    remainder="passthrough",
)

# --- 6. Define All Model Pipelines ---
# Note: Alphas for Ridge/Lasso are much smaller now,
# because the target variable is on a much smaller scale (log scale).
models = {
    "Linear Regression": Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
    ),
    # A smaller alpha is needed for the log-transformed target
    "Ridge Regression": Pipeline(
        steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=10.0))]
    ),
    "Lasso Regression": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Lasso(alpha=0.0005, max_iter=2000)),
        ]
    ),
    "K-Nearest Neighbors": Pipeline(
        steps=[("preprocessor", preprocessor), ("model", KNeighborsRegressor(n_neighbors=5))]
    ),
    "Decision Tree": Pipeline(
        steps=[("preprocessor", preprocessor), ("model", DecisionTreeRegressor(random_state=42))]
    ),
    "Bagging (Decision Tree)": Pipeline(
        steps=[("preprocessor", preprocessor), ("model", BaggingRegressor(random_state=42))]
    ),
    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    ),
    "AdaBoost": Pipeline(
        steps=[("preprocessor", preprocessor), ("model", AdaBoostRegressor(random_state=42))]
    ),
    "Gradient Boosting": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    ),
}
# We skip Polynomial Regression as it's too complex and slow

# --- 7. Run Cross-Validation for All Models ---
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

results = {}

for name, pipeline in models.items():
    print(f"Evaluating: {name}...")
    try:
        # We are now predicting y_log, not y
        scores = cross_val_score(
            pipeline, X, y_log, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1
        )
        # The result is now RMSLE (Root Mean Squared Logarithmic Error)
        results[name] = -scores.mean()
    except Exception as e:
        results[name] = f"Error: {e}"

print("\n" + "=" * 80 + "\n")
print("--- Performance Results (RMSLE on log-transformed HotelValue) ---")
print("Lower is better.\n")

sorted_results = sorted(
    results.items(),
    key=lambda item: (
        isinstance(item[1], float),
        -item[1] if isinstance(item[1], float) else 0,
    ),
    reverse=True,
)

for name, rmsle in sorted_results:
    if isinstance(rmsle, float):
        print(f"{name:<25}: {rmsle:.6f}")
    else:
        print(f"{name:<25}: {rmsle}")