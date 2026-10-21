import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES


# Calculate the four required regression metrics in one place.
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    return mae, mse, rmse, r2


# Create the outputs folder if it does not exist.
os.makedirs("outputs", exist_ok=True)   #import os is for this 

# Read the modelling table created in Part A.
modelling_table = pd.read_csv("data/modelling_table.csv")

# Using the 17 features in their proper order and predict delivery hours.
X = modelling_table[FEATURES]
y = modelling_table["delivery_hours"]

# Split the data into training and temporary parts first.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Split the temporary part equally into validation and test data.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Fit the scaler on training data only, then use it to transform train and test.
check_scaler = StandardScaler()
train_scaled = check_scaler.fit_transform(X_train)
test_scaled = check_scaler.transform(X_test)

train_scaled_max_abs_mean = float(np.abs(train_scaled.mean(axis=0)).max())
test_scaled_max_abs_mean = float(np.abs(test_scaled.mean(axis=0)).max())

# Save the required split and scaling checks.
checks = pd.DataFrame([
    {"metric": "train_rows", "value": len(X_train)},
    {"metric": "val_rows", "value": len(X_val)},
    {"metric": "test_rows", "value": len(X_test)},
    {"metric": "train_scaled_max_abs_mean", "value": train_scaled_max_abs_mean},
    {"metric": "test_scaled_max_abs_mean", "value": test_scaled_max_abs_mean},
])
checks.to_csv("outputs/part_b_checks.csv", index=False)


# Baseline: predict the training target mean for every validation shipment.
baseline_prediction = np.full(len(y_val), y_train.mean())
baseline_mae, baseline_mse, baseline_rmse, baseline_r2 = calculate_metrics(
    y_val, baseline_prediction
)

metric_rows = []
metric_rows.append({
    "model": "mean_baseline",
    "split": "val",
    "MAE": baseline_mae,
    "MSE": baseline_mse,
    "RMSE": baseline_rmse,
    "R2": baseline_r2,
})


# Test the required decision-tree depths on the same training and validation rows.
depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_results = []
tree_models = {}

for depth in depths:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_prediction))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_prediction))

    tree_models[depth] = tree
    tree_results.append({
        "max_depth": depth,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
    })

# Pick the lowest validation RMSE. If values tie, keep the first (smaller) depth.
best_tree_result = tree_results[0]
for result in tree_results[1:]:
    if result["val_rmse"] < best_tree_result["val_rmse"]:
        best_tree_result = result

best_depth = best_tree_result["max_depth"]

# Save None as the word 'unlimited' in the output CSV.
tree_depth_rows = []
for result in tree_results:
    if result["max_depth"] is None:
        depth_name = "unlimited"
    else:
        depth_name = result["max_depth"]

    tree_depth_rows.append({
        "max_depth": depth_name,
        "train_rmse": result["train_rmse"],
        "val_rmse": result["val_rmse"],
    })

pd.DataFrame(tree_depth_rows).round(4).to_csv(
    "outputs/part_b_tree_depth.csv", index=False
)


# Set up the four candidate models required in the project question.
models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "decision_tree": tree_models[best_depth],
    "random_forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
}

validation_rmse = {}

# Fit each candidate on training data and compare it on validation data.
for model_name, model in models.items():
    # The selected decision tree has already been fitted during the depth experiment.
    if model_name != "decision_tree":
        model.fit(X_train, y_train)

    val_prediction = model.predict(X_val)
    mae, mse, rmse, r2 = calculate_metrics(y_val, val_prediction)
    validation_rmse[model_name] = rmse

    metric_rows.append({
        "model": model_name,
        "split": "val",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    })


# Compare the three required learning rates for Gradient Boosting.
learning_rate_rows = []
for learning_rate in [0.01, 0.1, 1.0]:
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=learning_rate,
        random_state=42,
    )
    gb_model.fit(X_train, y_train)
    gb_prediction = gb_model.predict(X_val)

    gb_rmse = np.sqrt(mean_squared_error(y_val, gb_prediction))
    gb_r2 = r2_score(y_val, gb_prediction)

    learning_rate_rows.append({
        "learning_rate": learning_rate,
        "val_rmse": gb_rmse,
        "val_r2": gb_r2,
    })

pd.DataFrame(learning_rate_rows).round(4).to_csv(
    "outputs/part_b_gb_learning_rate.csv", index=False
)


# Rank the four candidates by validation RMSE and select the best two for CV.
validation_order = sorted(validation_rmse, key=validation_rmse.get)
top_two_models = validation_order[:2]

# Combine the training and validation rows for cross-validation (1,020 rows).
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

# Use the same five-fold split object for both selected models.
shared_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []

for model_name in top_two_models:
    cv_scores = cross_val_score(
        models[model_name],
        X_train_val,
        y_train_val,
        cv=shared_kfold,
    )

    cv_rows.append({
        "model": model_name,
        "mean_r2": cv_scores.mean(),
        "std_r2": cv_scores.std(),
    })

pd.DataFrame(cv_rows).round(4).to_csv(
    "outputs/part_b_cv.csv", index=False
)


# The model with the lowest validation RMSE is the final model.
#already fitted on training data, so evaluate it on test data only.
best_model_name = validation_order[0]

final_model = models[best_model_name]

test_prediction = final_model.predict(X_test)

test_mae, test_mse, test_rmse, test_r2 = calculate_metrics(y_test, test_prediction)

metric_rows.append({
    "model": best_model_name,
    "split": "test",
    "MAE": test_mae,
    "MSE": test_mse,
    "RMSE": test_rmse,
    "R2": test_r2,
})

# Save the baseline, four validation results and one final test result.
metrics = pd.DataFrame(metric_rows, columns=[
    "model", "split", "MAE", "MSE", "RMSE", "R2"
])
metrics.round(4).to_csv("outputs/part_b_metrics.csv", index=False)


# Print a short summary so the main checks are easy to see in PowerShell.
print("Train rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))
print("Train scaled max absolute mean:", train_scaled_max_abs_mean)
print("Test scaled max absolute mean:", test_scaled_max_abs_mean)

if best_depth is None:
    print("Best decision tree depth: unlimited")
else:
    print("Best decision tree depth:", best_depth)

print("Best model on validation:", best_model_name)
print("Validation RMSE:", validation_rmse[best_model_name])
print("Final test R2:", test_r2)



# Print only the most important results with proper understanding  for normal people.
print("\n========== QUICK SUMMARY ==========")
if best_depth is None:
    print("Best decision-tree depth: unlimited")
else:
    print("Best decision-tree depth:", best_depth)
print("Best model on validation:", best_model_name)
print(f"Validation RMSE: {validation_rmse[best_model_name]:.4f}")
print(f"Final test RMSE: {test_rmse:.4f}")
print(f"Final test R2: {test_r2 * 100:.2f}%")
print("Part B completed. All five CSV outputs saved in outputs.")
