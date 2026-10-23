
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (confusion_matrix,accuracy_score,f1_score,precision_score,recall_score,)

from src.shipment_prep import FEATURES, NOISE_COLUMNS


os.makedirs("outputs", exist_ok=True)
modelling_table = pd.read_csv("data/modelling_table.csv")

x = modelling_table[FEATURES]
y = modelling_table["is_delayed"].astype(int)


# Train-validation-test split using stratification in both steps.
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Train rows:", len(x_train))
print("Validation rows:", len(x_val))
print("Test rows:", len(x_test))
print("Delayed parcels - train:", int(y_train.sum()))
print("Delayed parcels - validation:", int(y_val.sum()))
print("Delayed parcels - test:", int(y_test.sum()))


# Calculate confusion-matrix values and the four required classification metrics.
def calculate_metrics(actual, predicted):
    tn, fp, fn, tp = confusion_matrix( actual, predicted, labels=[0, 1]).ravel()

    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


#applying the assignment's explicit rule: delayed when probability >= threshold.
def predict_at_threshold(model, features, threshold=0.5):
    probabilities = model.predict_proba(features)[:, 1]
    return (probabilities >= threshold).astype(int)


#create the blind baseline: predict every parcel as not delayed (0).
model_rows = []

blind_prediction = np.zeros(len(y_val), dtype=int)
blind_metrics = calculate_metrics(y_val, blind_prediction)

model_rows.append({
    "model": "blind_all_on_time",
    **blind_metrics,
})


#train Logistic Regression with StandardScaler in a Pipeline.
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

logistic_model.fit(x_train, y_train)
logistic_predictions = logistic_model.predict(x_val)

model_rows.append({
    "model": "logistic_regression",
    **calculate_metrics(y_val, logistic_predictions),
})


#try the required decision-tree depths and keep the highest validation F1.
tree_depths = [2, 3, 4, 5, 6, 8, 10]               ##depths are checked from smallest to largest, so ties keep the smaller depth.
best_tree_depth = None
best_tree_f1 = -1                              
best_tree_model = None

for depth in tree_depths:
    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42,
    )

    tree_model.fit(x_train, y_train)
    tree_predictions = tree_model.predict(x_val)
    tree_f1 = f1_score(y_val, tree_predictions, zero_division=0)

    if tree_f1 > best_tree_f1:
        best_tree_f1 = tree_f1
        best_tree_depth = depth
        best_tree_model = tree_model

model_rows.append({
    "model": f"decision_tree_depth_{best_tree_depth}",
    **calculate_metrics(y_val, best_tree_model.predict(x_val)),
})


# Train the default Random Forest using all 17 features.
default_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

default_forest_model.fit(x_train, y_train)
forest_predictions = default_forest_model.predict(x_val)

model_rows.append({
    "model": "random_forest_default",
    **calculate_metrics(y_val, forest_predictions),
})


# Save exactly the four required model rows and columns.
models_table = pd.DataFrame(
    model_rows,
    columns=[
        "model", "TN", "FP", "FN", "TP",
        "accuracy", "precision", "recall", "f1",
    ],
)

models_table.round(4).to_csv(
    "outputs/part_c_models.csv",
    index=False,
)


# Calculate one F-score per feature on the training rows.
all_features_selector = SelectKBest(
    score_func=f_classif,
    k="all",
)

all_features_selector.fit(x_train, y_train)
f_scores = all_features_selector.scores_


# Calculate Random Forest feature importance using the same training rows.
importance_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

importance_forest_model.fit(x_train, y_train)
rf_importances = importance_forest_model.feature_importances_


# Give every feature a unique rank: 1 means the highest F-score.
# If scores tie, the pinned FEATURES order breaks the tie consistently.
rank_order = np.argsort(-f_scores, kind="stable")

f_ranks = np.empty(len(FEATURES), dtype=int)
f_ranks[rank_order] = np.arange(1, len(FEATURES) + 1)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "f_rank": f_ranks,
    "rf_importance": rf_importances,
    "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES],
})

feature_scores.round(4).to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False,
)


# Compare SelectKBest using k = 5, 7, 10, 12 and 17 features.
selectk_rows = []

for k in [5, 7, 10, 12, 17]:
    selectk_model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )),
    ])

    selectk_model.fit(x_train, y_train)
    selectk_predictions = predict_at_threshold(selectk_model, x_val, 0.5)
    selectk_f1 = f1_score(
        y_val, selectk_predictions, zero_division=0
    )

    selectk_rows.append({
        "k": k,
        "val_f1": selectk_f1,
    })


# Save the SelectKBest comparison table.
pd.DataFrame(
    selectk_rows,
    columns=["k", "val_f1"],
).round(4).to_csv(
    "outputs/part_c_selectk.csv",
    index=False,
)


# Fit SelectKBest with k=7 on training rows only, then transform validation.
selector_7 = SelectKBest(score_func=f_classif, k=7)

x_train_7 = selector_7.fit_transform(x_train, y_train)
x_val_7 = selector_7.transform(x_val)


# These are the exact 27 combinations requested in the assignment.
parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}


# Run GridSearchCV. Default scoring for a classifier is accuracy.
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=parameter_grid,
    cv=5,
)

grid_search.fit(x_train_7, y_train)

grid_predictions = predict_at_threshold(
    grid_search.best_estimator_, x_val_7, 0.5
)
grid_val_f1 = f1_score(y_val, grid_predictions, zero_division=0)


# Run RandomizedSearchCV with the required nine candidates and seed.
randomized_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=parameter_grid,
    n_iter=9,
    cv=5,
    random_state=42,
)

randomized_search.fit(x_train_7, y_train)

randomized_predictions = predict_at_threshold(
    randomized_search.best_estimator_, x_val_7, 0.5
)
randomized_val_f1 = f1_score(
    y_val, randomized_predictions, zero_division=0
)


# Fit the untuned default forest on the same seven selected features.
default_forest_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=None,
    min_samples_leaf=1,
)

default_forest_7.fit(x_train_7, y_train)

default_7_predictions = predict_at_threshold(
    default_forest_7, x_val_7, 0.5
)
default_7_val_f1 = f1_score(
    y_val, default_7_predictions, zero_division=0
)


# Save three tuning rows. best_cv_accuracy is blank for the untuned forest.
tuning_rows = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1,
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(randomized_search.best_params_),
        "best_cv_accuracy": randomized_search.best_score_,
        "val_f1": randomized_val_f1,
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": default_7_val_f1,
    },
]

pd.DataFrame(
    tuning_rows,
    columns=["search", "candidates", "cv_folds", "total_fits","best_params", "best_cv_accuracy", "val_f1",]).round(4).to_csv("outputs/part_c_tuning.csv",index=False)


# Choose the candidate with the highest validation F1.

priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,                                       #default forest, then grid search, then randomized search.
    "randomized_search": 2,
}

winning_row = sorted(tuning_rows,key=lambda row: (-row["val_f1"], priority[row["search"]]))[0]

chosen_from = winning_row["search"]

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_
elif chosen_from == "randomized_search":
    chosen_params = randomized_search.best_params_
else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
    }


# Build the required final Pipeline and fit it on the training rows.
final_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(
        random_state=42,
        n_estimators=chosen_params["n_estimators"],
        max_depth=chosen_params["max_depth"],
        min_samples_leaf=chosen_params["min_samples_leaf"],
    )),
])

final_model.fit(x_train, y_train)


# Get validation probabilities and test each exact threshold from 0.1 to 0.9.
validation_probabilities = final_model.predict_proba(x_val)[:, 1]

threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

threshold_rows = []

for threshold in threshold_values:
    threshold_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    threshold_metrics = calculate_metrics(
        y_val, threshold_predictions
    )

    threshold_rows.append({
        "threshold": threshold,
        **threshold_metrics,
    })


pd.DataFrame(
    threshold_rows,
    columns=[
        "threshold", "TN", "FP", "FN", "TP",
        "accuracy", "precision", "recall", "f1",
    ],
).round(4).to_csv("outputs/part_c_thresholds.csv",index=False,)


# Use unrounded recall to apply the operations rule correctly.
eligible_thresholds = [
    row["threshold"]
    for row in threshold_rows
    if row["recall"] >= 0.80
]

if eligible_thresholds:
    chosen_threshold = max(eligible_thresholds)
else:
    chosen_threshold = 0.1

chosen_validation_metrics = next(
    row for row in threshold_rows
    if row["threshold"] == chosen_threshold
)


# Evaluate the final model on the test rows once, at the chosen threshold.
test_probabilities = final_model.predict_proba(x_test)[:, 1]
test_predictions = (test_probabilities >= chosen_threshold).astype(int)

test_metrics = calculate_metrics(y_test, test_predictions)

pd.DataFrame(
    [{
        "threshold": chosen_threshold,
        **test_metrics,
    }],
    columns=[
        "threshold", "TN", "FP", "FN", "TP",
        "accuracy", "precision", "recall", "f1",
    ],
).round(4).to_csv(
    "outputs/part_c_final_test.csv",
    index=False,
)


# Save the final model settings in one row. None means no maximum depth.
if chosen_params["max_depth"] is None:
    final_max_depth = "unlimited"
else:
    final_max_depth = chosen_params["max_depth"]

final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": final_max_depth,
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from,
}], columns=[
    "selector_k", "n_estimators", "max_depth",
    "min_samples_leaf", "threshold", "chosen_from",
])

final_config.to_csv(
    "outputs/part_c_final_config.csv",
    index=False,
)


# Print a short summary. Detailed result tables remain in the CSV files.
noise_importance_share = feature_scores.loc[
    feature_scores["is_noise"], "rf_importance"
].sum()

top_seven_features = (
    feature_scores.sort_values("f_rank")["feature"].head(7).tolist()
)

noise_outside_top_seven = bool(
    (
        feature_scores.loc[feature_scores["is_noise"], "f_rank"] >= 8
    ).all()
)

print("\n========== Part C Classification Summary ==========")
print("Best decision-tree depth:", best_tree_depth)
print("Blind baseline accuracy:", f"{blind_metrics['accuracy']:.4f}")
print(
    "Noise columns' combined forest importance:",
    f"{noise_importance_share:.4f}",
)
print(
    "All five noise columns outside top 7 by F-score:",
    noise_outside_top_seven,
)
print("Top 7 features by F-score:", ", ".join(top_seven_features))
print("Final configuration chosen from:", chosen_from)
print("Chosen alert threshold:", f"{chosen_threshold:.1f}")
print(
    "Validation precision / recall / F1:",
    f"{chosen_validation_metrics['precision']:.4f} / "
    f"{chosen_validation_metrics['recall']:.4f} / "
    f"{chosen_validation_metrics['f1']:.4f}",
)
print(
    "Final test precision / recall / F1:",
    f"{test_metrics['precision']:.4f} / "
    f"{test_metrics['recall']:.4f} / "
    f"{test_metrics['f1']:.4f}",
)
print("Part C completed. All seven CSV output files saved in outputs.")
