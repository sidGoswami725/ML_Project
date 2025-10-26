import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Suppress warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

print("--- Starting Random Forest Script ---")

# --- PHASE 1: LOAD DATA ---
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: train.csv or test.csv not found.")
    exit()

test_ids = df_test['Id']

# --- PHASE 2: PREPROCESSING AND FEATURE ENGINEERING ---
print("Starting preprocessing...")
df_train['HotelValue'] = np.log1p(df_train['HotelValue'])
y_train = df_train.pop('HotelValue')
all_data = pd.concat([df_train.drop('Id', axis=1), df_test.drop('Id', axis=1)], ignore_index=True)

for col in ['PoolQuality', 'ExtraFacility', 'FacadeType', 'BoundaryFence', 'LoungeQuality',
            'ParkingType', 'ParkingFinish', 'ParkingQuality', 'ParkingCondition',
            'BasementHeight', 'BasementCondition', 'BasementExposure',
            'BasementFacilityType1', 'BasementFacilityType2']:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

for col in all_data.columns:
    if all_data[col].dtype == "object":
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    else:
        all_data[col] = all_data[col].fillna(all_data[col].median())

all_data['HotelAge'] = all_data['YearSold'] - all_data['ConstructionYear']
all_data['YearsSinceRenovation'] = all_data['YearSold'] - all_data['RenovationYear']
all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
all_data['TotalBathrooms'] = (all_data['FullBaths'] + (0.5 * all_data['HalfBaths']) +
                              all_data['BasementFullBaths'] + (0.5 * all_data['BasementHalfBaths']))

all_data = pd.get_dummies(all_data, drop_first=True)
feature_names = all_data.columns.tolist()

X = all_data[:len(df_train)]
X_test = all_data[len(df_train):]
X_unscaled = X.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete.")
print(f"Final training features shape: {X_scaled.shape}")

# --- PHASE 3: MODEL TUNING (GridSearchCV) ---
print("\n--- Training Random Forest ---")
print("Running GridSearchCV (this may take a while)...")

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Define a parameter grid to search
# NOTE: This grid is kept small to run faster.
# For a full search, you would add more values.
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'max_features': ['sqrt', 0.5],
    'min_samples_leaf': [5, 10]
}

grid_rf = GridSearchCV(rf, param_grid, cv=10, 
                       scoring='neg_root_mean_squared_error', 
                       n_jobs=-1, verbose=1)

grid_rf.fit(X_scaled, y_train)

# --- PHASE 4: MODEL EVALUATION & ANALYSIS ---
print("\n--- Evaluation Results ---")
print(f"Best Parameters: {grid_rf.best_params_}")
print(f"Best CV RMSLE (log-scale): {abs(grid_rf.best_score_):.5f}")

# --- Feature Importance Analysis ---
print("\n--- Feature Importance Analysis ---")
# Re-train on UN SCALED data
best_params_rf = grid_rf.best_params_
rf_final = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params_rf)
rf_final.fit(X_unscaled, y_train)

importances = rf_final.feature_importances_
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

print("Top 10 Most Important Features:")
print(fi_df.head(10))

# --- PHASE 5: SUBMISSION ---
print("\n--- Generating Submission File ---")
predictions_log = grid_rf.predict(X_test_scaled)
predictions = np.expm1(predictions_log)

submission = pd.DataFrame({'Id': test_ids, 'HotelValue': predictions})
submission.to_csv('submission_random_forest.csv', index=False)

print("Successfully created 'submission_random_forest.csv'")