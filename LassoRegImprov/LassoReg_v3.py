import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import warnings
import time

# ==============================================================================
# --- Model: Lasso Regression (Final Zoom)
# --- Tuning: Manual loop with 10-fold CV and 'r2' scoring
# --- Preprocessing: User-provided script
# ==============================================================================

# Score got worse from 18959.588 to 19147.346. This means we are now overfitting 
# beyond alpha = 0.0006.
# This means v2 is our best script and alpha = 0.0006.

warnings.filterwarnings("ignore")
print("Starting Lasso Regression (Final Zoom) script...")

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

# --- PHASE 2: PREPROCESSING (Same as your best script) ---
df_train['HotelValue'] = np.log1p(df_train['HotelValue'])
y_train = df_train.pop('HotelValue')
all_data = pd.concat([df_train.drop('Id', axis=1), df_test.drop('Id', axis=1)], ignore_index=True)
print(f"Combined data shape: {all_data.shape}")
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
all_data['HotelAge'] = all_data['YearSold'] - all_data['ConstructionYear']
all_data['YearsSinceRenovation'] = all_data['YearSold'] - all_data['RenovationYear']
all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
all_data['TotalBathrooms'] = (all_data['FullBaths'] + (0.5 * all_data['HalfBaths']) +
                              all_data['BasementFullBaths'] + (0.5 * all_data['BasementHalfBaths']))
all_data = pd.get_dummies(all_data, drop_first=True)
X = all_data[:len(df_train)]
X_test = all_data[len(df_train):]
print(f"Final training features shape: {X.shape}")
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
# --- END OF PREPROCESSING ---

# --- PHASE 3: MODEL EVALUATION & SUBMISSION (Friend's Method) ---

print("\n" + "=" * 80)
print("Manually tuning Lasso (searching for peak R^2)...")

best_lasso_score = -float('inf')
best_lasso_alpha = 0

# Using a zoomed in list around alpha = 0.0006
alphas_lasso = [0.0005, 0.00051, 0.00052, 0.00053, 0.00054, 0.00055, 0.00056, 
                0.00057, 0.00058, 0.00059, 0.0006, 0.00061, 0.00062, 0.00063, 
                0.00064, 0.00065, 0.00066, 0.00067, 0.00068, 0.00069, 0.0007]

start_time = time.time()

for alpha in alphas_lasso:
    lasso = Lasso(alpha=alpha, max_iter=20000)
    
    cv_score = np.mean(cross_val_score(lasso, X, y_train, cv=10, scoring='r2', n_jobs=-1))
    
    # Use :.6f to show 6 decimal places for alpha
    print(f"Lasso (alpha={alpha:.6f}) - 10-Fold CV R^2 Score: {cv_score:.6f}")
    
    if cv_score > best_lasso_score:
        best_lasso_score = cv_score
        best_lasso_alpha = alpha

end_time = time.time()
print(f"\nLoop finished in {end_time - start_time:.2f} seconds.")
print("-" * 80)
print(f"Best Lasso Alpha found: {best_lasso_alpha:.6f} with R^2 Score: {best_lasso_score:.6f}")
print("-" * 80)


# 3.3 Train Final Model (using the best alpha found in the loop)
print(f"Training final model on all data using best alpha={best_lasso_alpha:.6f}...")
final_model = Lasso(alpha=best_lasso_alpha, max_iter=10000)
final_model.fit(X, y_train)

# 3.4 Generate Submission
print("Making predictions on test.csv...")
log_predictions = final_model.predict(X_test)
final_predictions = np.expm1(log_predictions)

submission_df = pd.DataFrame({"Id": test_ids, "HotelValue": final_predictions})
submission_df.to_csv("submission_lassov3.csv", index=False)

print("\n" + "=" * 80)
print("submission_lassov3.csv has been created!")
print(submission_df.head())
print("=" * 80)