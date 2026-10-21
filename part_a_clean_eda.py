import os
import pandas as pd
import numpy as np

from src.shipment_prep import clean_shipments, build_features, FEATURES

# Read the two input files.
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")


print("Raw rows:", len(raw))
print("Number of hubs:", len(hubs))

#  Keep a copy of the raw data after removing duplicates.
#  We use this copy to count missing values before filling them.
raw_unique = raw.drop_duplicates().copy()

# Count the data problems before cleaning/filling.
raw_rows = len(raw)
duplicate_rows_removed = len(raw) - len(raw_unique)
rows_after_dedup = len(raw_unique)

missing_distance_before = raw_unique["distance_km"].isna().sum()
missing_parcel_weight_before = raw_unique["parcel_weight_kg"].isna().sum()
missing_declared_value_before = raw_unique["declared_value"].isna().sum()
missing_payment_mode_before = raw_unique["payment_mode"].isna().sum()

weight_numeric = pd.to_numeric(raw_unique["parcel_weight_kg"], errors="coerce")
sentinel_weight_rows = (weight_numeric > 100).sum()

service_type_spellings_before = raw_unique["service_type"].nunique()

#  Clean the shipment data and save the historical medians.
clean, medians = clean_shipments(raw)
service_type_spellings_after = clean["service_type"].nunique()

# 5. Add hub information and create the features.
modelling_table = build_features(clean, hubs)

# 6. Create the delay target: 1 means late, 0 means not late.
modelling_table["is_delayed"] = (
    modelling_table["delivery_hours"] > modelling_table["promised_hours"]
).astype(int)

# 7. Save the modelling table for Parts B, C and D.
#    Create the output folder if it does not exist yet.
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
modelling_table.to_csv("data/modelling_table.csv", index=False)

print("Features:", len(FEATURES))
print("Medians:", medians)

# 8. Check for missing values across the 17 features and both target columns.
columns_to_check = FEATURES + ["delivery_hours", "is_delayed"]
nulls_in_features_and_target = modelling_table[columns_to_check].isna().sum().sum()


print("Missing distance:", missing_distance_before)
print("Missing weight:", missing_parcel_weight_before)
print("Missing declared value:", missing_declared_value_before)
print("Missing payment mode:", missing_payment_mode_before)
print("Invalid weight values:", sentinel_weight_rows)
print("Service types before:", service_type_spellings_before)
print("Service types after:", service_type_spellings_after)
print("Missing feature and target cells:", nulls_in_features_and_target)



# 9. Calculate the delivery-hours quartiles and IQR outlier bounds.
hours_q1 = modelling_table["delivery_hours"].quantile(0.25)
hours_q3 = modelling_table["delivery_hours"].quantile(0.75)
hours_iqr = hours_q3 - hours_q1
hours_lower_bound = hours_q1 - 1.5 * hours_iqr
hours_upper_bound = hours_q3 + 1.5 * hours_iqr
hours_outlier_count = ((modelling_table["delivery_hours"] < hours_lower_bound)|(modelling_table["delivery_hours"] > hours_upper_bound)).sum()

# 10. Calculate declared-value statistics from the filled modelling table.
declared_value_mean_inr = modelling_table["declared_value_inr"].mean()
declared_value_median_inr = modelling_table["declared_value_inr"].median()
log_declared_value_mean = modelling_table["log_declared_value"].mean()
log_declared_value_median = modelling_table["log_declared_value"].median()

delayed_count = modelling_table["is_delayed"].sum()
delay_rate = modelling_table["is_delayed"].mean()

# 11. Put all required checks into a table.
checks = [
    {"metric": "raw_rows", "value": raw_rows},
    {"metric": "duplicate_rows_removed", "value": duplicate_rows_removed},
    {"metric": "rows_after_dedup", "value": rows_after_dedup},
    {"metric": "missing_distance_km_before", "value": missing_distance_before},
    {"metric": "missing_parcel_weight_before", "value": missing_parcel_weight_before},
    {"metric": "missing_declared_value_before", "value": missing_declared_value_before},
    {"metric": "missing_payment_mode_before", "value": missing_payment_mode_before},
    {"metric": "sentinel_weight_rows", "value": sentinel_weight_rows},
    {"metric": "service_type_spellings_before", "value": service_type_spellings_before},
    {"metric": "service_type_spellings_after", "value": service_type_spellings_after},
    {"metric": "median_distance_km", "value": round(medians["distance_km"], 4)},
    {"metric": "median_parcel_weight_kg", "value": round(medians["parcel_weight_kg"], 4)},
    {"metric": "median_declared_value_inr", "value": round(medians["declared_value_inr"], 4)},
    {"metric": "rows_after_hub_merge", "value": len(modelling_table)},
    {"metric": "nulls_in_features_and_target", "value": int(nulls_in_features_and_target)},
    {"metric": "delayed_count", "value": int(delayed_count)},
    {"metric": "delay_rate", "value": round(delay_rate, 4)},
    {"metric": "hours_q1", "value": round(hours_q1, 4)},
    {"metric": "hours_q3", "value": round(hours_q3, 4)},
    {"metric": "hours_lower_bound", "value": round(hours_lower_bound, 4)},
    {"metric": "hours_upper_bound", "value": round(hours_upper_bound, 4)},
    {"metric": "hours_outlier_count", "value": int(hours_outlier_count)},
    {"metric": "declared_value_mean_inr", "value": round(declared_value_mean_inr, 4)},
    {"metric": "declared_value_median_inr", "value": round(declared_value_median_inr, 4)},
    {"metric": "log_declared_value_mean", "value": round(log_declared_value_mean, 4)},
    {"metric": "log_declared_value_median", "value": round(log_declared_value_median, 4)}
]

pd.DataFrame(checks, columns=["metric", "value"]).to_csv(
    "outputs/part_a_checks.csv", index=False
)

# Make the delay-by-group table for service, hub tier and distance band.
group_rows = []

for service, group in modelling_table.groupby("service_type"):
    group_rows.append({
        "dimension": "service_type",
        "group": service,
        "shipments": len(group),
        "delayed": int(group["is_delayed"].sum()),
        "delay_rate": round(group["is_delayed"].mean(), 4),
        "mean_hours": round(group["delivery_hours"].mean(), 4)
    })

for tier, group in modelling_table.groupby("hub_tier"):
    group_rows.append({
        "dimension": "hub_tier",
        "group": str(int(tier)),
        "shipments": len(group),
        "delayed": int(group["is_delayed"].sum()),
        "delay_rate": round(group["is_delayed"].mean(), 4),
        "mean_hours": round(group["delivery_hours"].mean(), 4)
    })

# The right edge of every distance band is included.
distance_groups = modelling_table.copy()
distance_groups["distance_band"] = pd.cut(
    distance_groups["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True
)

for band, group in distance_groups.groupby("distance_band", observed=False):
    group_rows.append({
        "dimension": "distance_band",
        "group": str(band),
        "shipments": len(group),
        "delayed": int(group["is_delayed"].sum()),
        "delay_rate": round(group["is_delayed"].mean(), 4),
        "mean_hours": round(group["delivery_hours"].mean(), 4)
    })

pd.DataFrame(
    group_rows,
    columns=["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]).to_csv("outputs/part_a_delay_by_group.csv", index=False)

# Save a complete correlation matrix for all numeric columns.
correlation_matrix = modelling_table.corr(numeric_only=True).round(4)
correlation_matrix.to_csv("outputs/part_a_correlations.csv", index=True, index_label="column")

#Print a short summary so the run can be checked in the terminal.
print("Raw rows:", raw_rows)
print("Duplicate rows removed:", duplicate_rows_removed)
print("Rows after cleaning:", len(clean))
print("Rows after hub merge:", len(modelling_table))
print("Delayed shipments:", int(delayed_count))
print("Delay rate:", round(delay_rate, 4))
print("Missing cells in features/target:", int(nulls_in_features_and_target))
print("Part A completed. CSV outputs saved.")
