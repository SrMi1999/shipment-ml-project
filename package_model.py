import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)  #if there is not folder it will make 
os.makedirs("outputs", exist_ok=True)

#read the final model setting from part C
final_config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
selector_k = int(final_config["selector_k"])
n_estimators = int(final_config["n_estimators"])
min_samples_leaf = int(final_config["min_samples_leaf"])

if pd.isna(final_config["max_depth"]) or str(final_config["max_depth"]).lower() in ["none", "unlimited", "nan"]:
    max_depth = None
else:
    max_depth = int(final_config["max_depth"])

threshold = float(final_config["threshold"])

#Now loading the modelling dataset-
modelling_table = pd.read_csv("data/modelling_table.csv")
X = modelling_table[FEATURES]
y = modelling_table["is_delayed"].astype(int)

#making the same stratified train, validation and test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

#building the final classification pipeline
final_model = Pipeline([
    (
        "selector",
        SelectKBest(score_func=f_classif, k=selector_k)
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    )
])

#train the final model using training data
final_model.fit(X_train, y_train)
joblib.dump(final_model, "models/delay_model.pkl")  #saving the trained model

#save the selected probability threshold
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
#save the feature order use by the model
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

#calculate an save the training data imputation values
raw_shipments = pd.read_csv("data/raw_shipments.csv")
cleaned_shipments, medians = clean_shipments(raw_shipments)
impute_values = pd.DataFrame([
    {
        "column": column,
        "median": median
    }
    for column, median in medians.items()
])
impute_values.to_csv("models/impute_values.csv", index=False)

#reload the save model
reloaded_model = joblib.load("models/delay_model.pkl")
#comapre the predictions from the orginal and reload model
original_probabilities = final_model.predict_proba(X_test)[:, 1]
reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]
reload_max_abs_probability_diff = float(np.max(np.abs(original_probabilities - reloaded_probabilities)))


#save the reload verfications result
reload_checks = pd.DataFrame([
    {
        "metric": "test_rows_scored",
        "value": len(X_test)
    },
    {
        "metric": "reload_max_abs_probability_diff",
        "value": reload_max_abs_probability_diff
    },
    {
        "metric": "threshold",
        "value": threshold
    }
])

reload_checks.to_csv("outputs/part_d_reload_check.csv", index=False)

#now print the summary
print("\n========== MODEL PACKAGING SUMMARY ==========")

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))
print("Selected features:", selector_k)
print("Probability threshold:", threshold)
print("Reload probability difference:", reload_max_abs_probability_diff)
print("Model saved: models/delay_model.pkl")
print("Model packaging completed.")





