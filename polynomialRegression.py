import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import warnings

# ==============================================================================
# --- Model: Polynomial Regression
# --- Tuning: None (Degree=2)
# --- Preprocessing and Feature Engineering (Modified for Poly)
# ==============================================================================

warnings.filterwarnings("ignore")
print("Starting Polynomial Regression script...")

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
train_ids = df_train['Id']

# --- PHASE 2: PREPROCESSING AND FEATURE ENGINEERING ---

# 2.1 Log Transform Target Variable
df_train['HotelValue'] = np.log1p(df_train['HotelValue'])

# 2.2 Combine train and test data for consistent preprocessing
y_train = df_train.pop('HotelValue')
all_data = pd.concat([df_train.drop('Id', axis=1), df_test.drop('Id', axis=1)], ignore_index=True)
print(f"Combined data shape: {all_data.shape}")

# 2.3 Handle Missing Values
for col in ['PoolQuality', 'ExtraFacility', 'FacadeType', 'BoundaryFence', 'LoungeQuality',
            'ParkingType', 'ParkingFinish', 'ParkingQuality', 'ParkingCondition',
            'BasementHeight', 'BasementCondition', 'BasementExposure',
            'BasementFacilityType1', 'BasementFacilityType2']:
    all_data[col] = all_data[col].fillna('None')

for col in all_data.columns:
    if all_data[col].dtype == "object":
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    else:
        all_data[col] = all_data[col].fillna(all_data[col].median())

print(f"Missing values remaining: {all_data.isnull().sum().sum()}")

# 2.4 Feature Engineering
all_data['HotelAge'] = all_data['YearSold'] - all_data['ConstructionYear']
all_data['YearsSinceRenovation'] = all_data['YearSold'] - all_data['RenovationYear']
all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
all_data['TotalBathrooms'] = (all_data['FullBaths'] + (0.5 * all_data['HalfBaths']) +
                              all_data['BasementFullBaths'] + (0.5 * all_data['BasementHalfBaths']))

# 2.5 Encode Categorical Features
all_data = pd.get_dummies(all_data, drop_first=True)

# 2.6 Separate back into training and testing sets
X = all_data[:len(df_train)]
X_test = all_data[len(df_train):]

print(f"Final (pre-poly) training features shape: {X.shape}")
print(f"Final (pre-poly) test features shape: {X_test.shape}")

# 2.7 ** MODIFIED STEP FOR POLYNOMIAL **
# We must create polynomial features *before* scaling.
# We use a Pipeline for this to keep it clean.
# WARNING: Degree=2 will create thousands of features and be slow.
print("Applying PolynomialFeatures (Degree=2) and Scaling...")
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

X = poly_pipeline.fit_transform(X)
X_test = poly_pipeline.transform(X_test)
print(f"Post-poly training features shape: {X.shape}")

# --- PHASE 3: MODEL EVALUATION & SUBMISSION ---

# 3.1 Define Model
model = LinearRegression()

# 3.2 Evaluate Model
print("Evaluating Polynomial Regression (Degree=2) using 5-fold CV...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    model, X, y_train, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1
)
print(f"Average RMSLE (CV Score): {-cv_scores.mean():.6f}")
print("-" * 80)

# 3.3 Train Final Model
print("Training final model on all data (This may be slow)...")
model.fit(X, y_train)

# 3.4 Generate Submission
print("Making predictions on test.csv...")
log_predictions = model.predict(X_test)
final_predictions = np.expm1(log_predictions)

submission_df = pd.DataFrame({"Id": test_ids, "HotelValue": final_predictions})
submission_df.to_csv("submission_poly.csv", index=False)

print("\n" + "=" * 80)
print("submission_poly.csv has been created!")
print(submission_df.head())
print("=" * 80)