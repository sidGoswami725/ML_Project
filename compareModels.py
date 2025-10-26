import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
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

# --- 2. Define Target (y) and Features (X) ---
# Using the original, skewed target variable
y = df["HotelValue"]
X = df.drop(["HotelValue", "Id"], axis=1)

# --- 3. Clean Up Columns ---
# These columns have too many missing values (based on our EDA)
cols_to_drop = ["PoolQuality", "ExtraFacility", "ServiceLaneType", "BoundaryFence"]
X = X.drop(columns=cols_to_drop)

# Identify numerical and categorical features
numerical_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(include="object").columns

print(f"Running evaluation on {X.shape[1]} features.")
print("Target variable: 'HotelValue' (original, skewed)")
print("Metric: RMSE (Lower is better)\n")
print("=" * 80 + "\n")

# --- 4. Create Standard Preprocessing Pipelines ---

# Standard Numerical pipeline
# Step 1: Impute missing values with the median
# Step 2: Scale features
num_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Categorical pipeline
# Step 1: Impute missing values with the most frequent value
# Step 2: One-hot encode (convert categories to numbers)
cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Standard preprocessor
# This ColumnTransformer applies the correct pipeline to the correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols),
    ],
    remainder="passthrough",  # Keep any columns not specified
)

# --- 5. Define All Model Pipelines ---

# A dictionary to hold all our model pipelines
models = {
    "Linear Regression": Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
    ),
    "Ridge Regression": Pipeline(
        steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=1.0))]
    ),
    "Lasso Regression": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Lasso(alpha=100.0, max_iter=2000)),
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

# --- Special Case: Polynomial Regression ---
# For Polynomial Regression, features must be created *before* scaling.
# We create a special numerical pipeline for it.
num_pipeline_poly = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),  # Create x^2, x*y features
        ("scaler", StandardScaler()),  # Then scale them
    ]
)

# New preprocessor using the poly-numerical pipeline
preprocessor_poly = ColumnTransformer(
    transformers=[
        ("num", num_pipeline_poly, numerical_cols),
        ("cat", cat_pipeline, categorical_cols),
    ],
    remainder="passthrough",
)

# Add to our models dictionary
models["Polynomial Regression"] = Pipeline(
    steps=[("preprocessor", preprocessor_poly), ("model", LinearRegression())]
)

# --- 6. Run Cross-Validation for All Models ---

# Suppress warnings, especially ConvergenceWarning for Lasso/Poly
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

results = {}

for name, pipeline in models.items():
    print(f"Evaluating: {name}...")
    try:
        # Use 5-fold cross-validation to get a robust RMSE score
        # n_jobs=-1 uses all available CPU cores to speed up
        scores = cross_val_score(
            pipeline, X, y, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1
        )
        results[name] = -scores.mean()
    except Exception as e:
        results[name] = f"Error: {e}"

print("\n" + "=" * 80 + "\n")
print("--- Performance Results (RMSE on original HotelValue) ---")
print("Lower is better.\n")

# Sort results by RMSE, best to worst
# This handles potential "Error" strings
sorted_results = sorted(
    results.items(),
    key=lambda item: (
        isinstance(item[1], float),
        -item[1] if isinstance(item[1], float) else 0,
    ),
    reverse=True,
)

for name, rmse in sorted_results:
    if isinstance(rmse, float):
        print(f"{name:<25}: {rmse:,.2f}")
    else:
        print(f"{name:<25}: {rmse}")

print(
    "\nNote: 'Bayes Classifier' and 'Generative Modelling' were skipped as they are classification methods, not regression."
)