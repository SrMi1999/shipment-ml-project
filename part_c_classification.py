import os
import numpy as np
import pandas as pd

from sklearn import pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from src.shipment_prep import FEATURES, NOISE_COLUMNS


os.makedirs("outputs", exist_ok=True)  # Create the outputs folder if it does not exist.
modelling_table = pd.read_csv("data/modelling_table.csv")


x= modelling_table[FEATURES]
y= modelling_table["is_delayed"].astype(int)

#train-test-validation split
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("Train rows:", len(x_train))
print("Validation rows:", len(x_val))
print("Test rows:", len(x_test))
print("Delayed parcels - train:", int(y_train.sum()))
print("Delayed parcels - validation:", int(y_val.sum()))
print("Delayed parcels - test:", int(y_test.sum()))


#Calculate confusion-matrix values and the four required classification metrics.
def calculate_metrics(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1= f1_score(actual, predicted, zero_division=0)
    
    return {"TN": int(tn),
            "FP": int(fp), 
            "FN": int(fn), 
            "TP": int(tp), 
            "accuracy": accuracy, 
            "precision": precision, 
            "recall": recall, 
            "f1": f1
    }
    
#Create the blind baseline: predict every parcel as not delayed (0)
model_rows = []
blind_prediction = np.zeros(len(y_val), dtype=int)
blind_metrics = calculate_metrics(y_val, blind_prediction)
model_rows.append({"model": "blind_all_ontime", "metrics": blind_metrics})

#Train Logistic Regression with StandardScaler in a Pipeline.
logistic_model = Pipeline([("scaler", StandardScaler()),("model", LogisticRegression())])

logistic_model.fit(x_train, y_train)
logistic_predictions = logistic_model.predict(x_val)

model_rows.append({"model": "logistic_regression", "metrics": calculate_metrics(y_val, logistic_predictions)})

#Try the required decision-tree depths and keep the highest validation F1.
tree_depths = [2, 3, 4, 5, 6, 8, 10]
best_tree_depth = None
best_tree_f1 = -1
best_tree_model = None

for depth in tree_depths:
    tree_model = DecisionTreeClassifier(criterion="gini",max_depth=depth,random_state=42)
    tree_model.fit(x_train, y_train)
    tree_predictions = tree_model.predict(x_val)
    tree_f1 = f1_score(y_val, tree_predictions, zero_division=0)
    
    
    if tree_f1 > best_tree_f1:
        best_tree_f1 = tree_f1
        best_tree_depth = depth
        best_tree_model = tree_model
        
model_rows.append({"model": f"decision_tree_depth_{best_tree_depth}", 
                   "metrics": calculate_metrics(y_val, best_tree_model.predict(x_val))
                   })           


#Train the default Random Forest using all 17 features.

default_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
default_forest_model.fit(x_train, y_train)
forest_predictions = default_forest_model.predict(x_val)

model_rows.append({"model": "random_forest_default", "metrics": calculate_metrics(y_val, forest_predictions)})

# Save the exactly the four required model rows and columns.
models_table = pd.DataFrame(model_rows, columns=[ "model", "TN", "FP", "FN", "TP","accuracy", "precision", "recall", "f1"])

models_table.round(4).to_csv("outputs/part_c_classification.csv", index=False)


#Calculate one F-score per feature on the training rows.
all_features_selector = SelectKBest(score_func=f_classif, k="all")
all_features_selector.fit(x_train, y_train)

f_scores = all_features_selector.scores_

#calculate Random Forest feature importance using the same training rows.

importance_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
importance_forest_model.fit(x_train, y_train)
rf_importances = importance_forest_model.feature_importances_

#give every feature a unique rank: 1 means the highest F-score.
rank_order = np.argsort(-f_scores, kind="stable")

f_ranks = np.empty(len(FEATURES), dtype=int)
                                                         #If two scores tie, the pinned FEATURES order breaks the tie consistently.
f_ranks[rank_order] = np.arange(1, len(FEATURES) + 1)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "f_rank": f_ranks,
    "rf_importance": rf_importances,
    "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES],
})

feature_scores.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False) #output csv file part_c_feature_scores.csv.

#compare select SelectKBest using k = 5, 7, 10, 12 and 17 features.
selectk_rows = []

for k in [5, 7, 10, 12, 17]:
    selectk_model = pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("model", RandomForestClassifier( n_estimators=100,random_state=42,)),
    ])
    selectk_model.fit(x_train, y_train)
    selectk_predictions = selectk_model.predict(x_val)
    selectk_f1 = f1_score(y_val, selectk_predictions, zero_division=0)
    
    
    selectk_rows.append({"k": k, "f1": selectk_f1})
    
#output part_c_selectK.csv file    
pd.DataFrame(selectk_rows, columns=["k", "val_f1"]).round(4).to_csv("outputs/part_c_selectk.csv", index=False)
    
    
#Fit SelectKBest with k=7 on training rows only, then transform validation rows.  



    
    
    
    



