# ML_Project

This repository contains the code and results for a machine learning project focused on predicting hotel property values based on a given dataset, as part of a Kaggle competition checkpoint.

## Project Goal

The primary objective was to train, evaluate, and tune various regression models to accurately predict the `HotelValue` for properties listed in the `test.csv` dataset, aiming for the best possible score on the Kaggle public leaderboard. A secondary goal was to explore different modeling techniques, preprocessing strategies, and hyperparameter tuning methods, documenting the findings as required by the course TAs.

## Dataset

* **`train.csv`**: Contains 1200 property records with 79 features and the target variable `HotelValue`.
* **`test.csv`**: Contains 260 property records with the same 79 features, used for generating predictions.
* **Target Variable (`HotelValue`)**: Showed significant right-skewness, necessitating a log transformation (`np.log1p`) before modeling.
* **Features**: Included numerical (size, age, counts), categorical (location, type, quality ratings), requiring careful preprocessing. Many features had missing values.

## Methodology

1.  **Preprocessing**:
    * **Log Transformation**: Applied `np.log1p` to `HotelValue`.
    * **Imputation**: Handled missing values strategically ('None' for absent features, median for numerical, mode for categorical).
    * **Feature Engineering**: Created new features like `TotalSF`, `HotelAge`, `YearsSinceRenovation`, `TotalBathrooms`.
    * **Encoding**: Primarily used **One-Hot Encoding** (`pd.get_dummies`) resulting in 262 features. An alternative pipeline using **Ordinal Encoding** (with Target Encoding and Feature Selection) was explored specifically for Gradient Boosting.
    * **Scaling**: Applied `StandardScaler` to all final features.
2.  **Model Training & Tuning**:
    * Implemented 11 different regression models (Linear, Polynomial, Ridge, Lasso, KNN, Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost).
    * Used `GridSearchCV` and manual loops with cross-validation (5-fold, 10-fold, `RepeatedKFold`) to find optimal hyperparameters.
    * Tuning was iteratively guided by **Kaggle leaderboard feedback** due to observed discrepancies between local CV scores and test set performance.
3.  **Evaluation**:
    * Local performance measured using RMSLE (Root Mean Squared Logarithmic Error) via cross-validation.
    * Final performance measured based on scores obtained on the Kaggle public leaderboard.

## Models Implemented & Results

The following models were trained and evaluated. The table summarizes the best Kaggle score achieved for each approach after tuning:

| Model Name             | Best Kaggle Score | Notes                                                                                       |
| :--------------------- | :---------------- | :------------------------------------------------------------------------------------------ |
| **Ridge Regression** | **18,322.343** | **Champion Model.** Best generalization achieved with `alpha=5` after iterative tuning.     |
| Linear Regression    | 18,664.752        | Strong baseline using OneHot preprocessing.                                                 |
| Lasso Regression     | 18,959.588        | Best score via manual R² tuning (`alpha=0.0006`). Less stable than Ridge.                 |
| Gradient Boosting    | 21,554.736        | Best GB score required Ordinal/TE/Jitter (v3). OneHot preprocessing overfit badly (~30k). |
| Polynomial Regression| 26,210.063        | Overfit severely. Ridge regularization didn't improve Kaggle score.                     |
| XGBoost                | 29,708.691        | Overfit badly on Kaggle with OneHot preprocessing. Regularization didn't help.        |
| Bagging Regressor    | 32,036.205        | Beat Decision Tree but overfit compared to linear models (using OneHot).                |
| Random Forest        | 35,120.114        | Significant overfitting on Kaggle despite strong CV (using OneHot).                      |
| AdaBoost               | 39,176.737        | Overfit significantly on Kaggle (using OneHot).                                             |
| K-Nearest Neighbors  | 41,481.585        | Poor generalization, likely due to high dimensionality (262 features).                    |
| Decision Tree        | 41,583.308        | Worst Kaggle score, confirming instability and overfitting.                                 |

**Key Finding**: Simple, well-regularized linear models (Ridge) demonstrated superior generalization compared to complex tree-based ensembles, which tended to overfit the training data despite strong local CV scores. Careful, leaderboard-guided tuning was essential.

## Repository Structure

* `/` (root):
    * Contains the initial Python scripts for each of the 11 models (e.g., `RidgeRegression.py`, `GradientBoosting.py`).
    * Contains the initial submission `.csv` files corresponding to these scripts.
    * Contains `train.csv`, `test.csv`, and `sample_submission.csv`.
* `/RidgeRegImprov/`: Contains iterative tuning scripts for Ridge Regression (`v1` to `v5`) and their corresponding submission `.csv` files.
* `/LassoRegImprov/`: Contains iterative tuning scripts for Lasso Regression (`v1` to `v3`, `friend_method`) and their corresponding submission `.csv` files.
* `/PolyRegImprov/`: Contains tuning scripts for Polynomial Regression (`v1`, `v2`) and their corresponding submission `.csv` files.
* `/GradBoostImprov/`: Contains alternative/tuned Gradient Boosting scripts (`v1` to `v3`, `stochastic`) and their corresponding submission `.csv` files.
* `README.md`: This file.

*(Note: Add/modify subfolder names like `/XGBoostImprov/` if you performed iterative tuning for other models)*

## How to Run

1.  Clone the repository.
2.  Ensure you have Python and necessary libraries installed (pandas, numpy, scikit-learn, xgboost).
    ```bash
    pip install pandas numpy scikit-learn xgboost
    ```
3.  The `train.csv` and `test.csv` files are in the root directory.
4.  Navigate to the desired folder (root or a subfolder like `RidgeRegImprov`).
5.  Run the chosen Python script (e.g., `python ../RidgeReg_v4.py` if running from root, or `python RidgeReg_v4.py` if inside the subfolder).
6.  The script will perform preprocessing, tuning (if applicable), train the final model, and save the submission file (usually within the same folder or potentially a specified output path).
