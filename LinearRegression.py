import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings

# ==============================================================================
# --- Model: Linear Regression
# --- Tuning: None
# --- Preprocessing and Feature Engineering
# ==============================================================================

warnings.filterwarnings("ignore")
print("Starting Linear Regression script...")

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
train_ids = df_train['Id'] # Keep for splitting

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

print(f"Final training features shape: {X.shape}")
print(f"Final test features shape: {X_test.shape}")

# 2.7 Scale Numerical Features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# --- PHASE 3: MODEL EVALUATION & SUBMISSION ---

# 3.1 Define Model
model = LinearRegression()

# 3.2 Evaluate Model (No Tuning)
print("Evaluating Linear Regression using 5-fold CV...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    model, X, y_train, cv=kf, scoring="neg_root_mean_squared_error"
)
print(f"Average RMSLE (CV Score): {-cv_scores.mean():.6f}")
print("-" * 80)

# 3.3 Train Final Model
print("Training final model on all data...")
model.fit(X, y_train)

# 3.4 Generate Submission
print("Making predictions on test.csv...")
log_predictions = model.predict(X_test)
final_predictions = np.expm1(log_predictions)

submission_df = pd.DataFrame({"Id": test_ids, "HotelValue": final_predictions})
submission_df.to_csv("submission_lr.csv", index=False)

print("\n" + "=" * 80)
print("submission_lr.csv has been created!")
print(submission_df.head())
print("=" * 80)