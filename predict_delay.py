import os
import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)
new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

model = joblib.load("models/delay_model.pkl")
threshold = pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0]           #loading the save model and supporting files
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute_values = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute_values["column"], impute_values["median"]))

# cleaning the new shipments using the training median
cleaned_shipments, _ = clean_shipments(new_shipments, medians=medians)


#Build the same features used during model training

new_features = build_features(cleaned_shipments, hubs)

X_new = new_features.reindex(columns=feature_order)      #Arrange the features in the saved training order


delay_probabilities = model.predict_proba(X_new)[:, 1]   #Predict the probability of delay
flagged = (delay_probabilities >= threshold).astype(int) #apply the saved threshold

#add prediction results to the shipment data
new_features["delay_probability"] = np.round(delay_probabilities, 4)
new_features["flagged"] = flagged

#Load the hub segments
hub_segments = pd.read_csv("outputs/hub_segments.csv")

new_features = new_features.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")

#assign an operational action to each shipment
new_features["action"] = np.where(new_features["flagged"] == 0, "No action",
    np.where(new_features["segment"] == "Escalate", "Escalate to hub manager", "Proactive customer call"))


#Keep only the required output columns
new_shipment_actions = new_features[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
].copy()



new_shipment_actions.to_csv("outputs/new_shipment_actions.csv", index=False) #finally save the final shipment action file

#now final printing the important results
print("\n========== NEW SHIPMENT PREDICTION SUMMARY ==========")

print("New shipments processed:", len(new_shipment_actions))
print("Flagged shipments:", int(new_shipment_actions["flagged"].sum()))
print("Hub manager escalations:",int(((new_shipment_actions["flagged"] == 1)&(new_shipment_actions["action"] == "Escalate to hub manager")).sum()))
print("Proactive customer calls:", int((new_shipment_actions["action"] == "Proactive customer call").sum()))
print("No action:",int((new_shipment_actions["action"] == "No action").sum()))
print("Output saved: outputs/new_shipment_actions.csv")
print("New shipment prediction completed.")










