import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Suppress warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

print("--- Starting Decision Tree Script ---")

# --- PHASE 1: LOAD DATA ---
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: train.csv or test.csv not found.")
    exit()

# Store IDs for submission
test_ids = df_test['Id']

# --- PHASE 2: PREPROCESSING AND FEATURE ENGINEERING ---
print("Starting preprocessing...")

# 2.1 Log Transform Target
df_train['HotelValue'] = np.log1p(df_train['HotelValue'])
y_train = df_train.pop('HotelValue')

# 2.2 Combine
all_data = pd.concat([df_train.drop('Id', axis=1), df_test.drop('Id', axis=1)], ignore_index=True)

# 2.3 Handle Missing Values
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

# 2.4 Feature Engineering
all_data['HotelAge'] = all_data['YearSold'] - all_data['ConstructionYear']
all_data['YearsSinceRenovation'] = all_data['YearSold'] - all_data['RenovationYear']
all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
all_data['TotalBathrooms'] = (all_data['FullBaths'] + (0.5 * all_data['HalfBaths']) +
                              all_data['BasementFullBaths'] + (0.5 * all_data['BasementHalfBaths']))

# 2.5 Encode Categorical Features
all_data = pd.get_dummies(all_data, drop_first=True)
feature_names = all_data.columns.tolist() # Capture feature names

# 2.6 Separate
X = all_data[:len(df_train)]
X_test = all_data[len(df_train):]
X_unscaled = X.copy() # Save unscaled data for feature importance

# 2.7 Scale Numerical Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete.")
print(f"Final training features shape: {X_scaled.shape}")

# --- PHASE 3: MODEL TUNING (GridSearchCV) ---
print("\n--- Training Decision Tree ---")
print("Running GridSearchCV (this may take a moment)...")

dt = DecisionTreeRegressor(random_state=42)

# Define a parameter grid to search
param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_leaf': [5, 10, 20],
    'min_samples_split': [10, 20, 40]
}

# Use 5-fold CV to find the best parameters
grid_dt = GridSearchCV(dt, param_grid, cv=10, 
                       scoring='neg_root_mean_squared_error', 
                       n_jobs=-1, verbose=1)

grid_dt.fit(X_scaled, y_train)

# --- PHASE 4: MODEL EVALUATION & ANALYSIS ---
print("\n--- Evaluation Results ---")
print(f"Best Parameters: {grid_dt.best_params_}")
# We print the absolute value since the score is negative
print(f"Best CV RMSLE (log-scale): {abs(grid_dt.best_score_):.5f}")

# --- Feature Importance Analysis ---
print("\n--- Feature Importance Analysis ---")
# Re-train the best model on UN SCALED data to get interpretable importances
best_params_dt = grid_dt.best_params_
dt_final = DecisionTreeRegressor(random_state=42, **best_params_dt)
dt_final.fit(X_unscaled, y_train)

importances = dt_final.feature_importances_
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

print("Top 10 Most Important Features:")
print(fi_df.head(10))

# --- PHASE 5: SUBMISSION ---
print("\n--- Generating Submission File ---")
# Use the model trained on SCALED data (grid_dt) for prediction
predictions_log = grid_dt.predict(X_test_scaled)
predictions = np.expm1(predictions_log) # Reverse the log transform

submission = pd.DataFrame({'Id': test_ids, 'HotelValue': predictions})
submission.to_csv('submission_decision_tree.csv', index=False)

print("Successfully created 'submission_decision_tree.csv'")