import numpy as np
import pandas as pd


PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
]


def clean_shipments(raw, medians=None):
    clean_df = raw.copy()
    clean_df = clean_df.drop_duplicates()

    clean_df["service_type"] = clean_df["service_type"].str.strip().str.title()
    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")

    clean_df["declared_value"] = (
        clean_df["declared_value"].astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    clean_df["declared_value_inr"] = pd.to_numeric(
        clean_df["declared_value"], errors="coerce"
    )
    clean_df = clean_df.drop(columns=["declared_value"])

    clean_df["dispatch_time"] = pd.to_datetime(clean_df["dispatch_time"])
    clean_df["distance_km"] = pd.to_numeric(clean_df["distance_km"], errors="coerce")
    clean_df["parcel_weight_kg"] = pd.to_numeric(clean_df["parcel_weight_kg"], errors="coerce")
    clean_df.loc[clean_df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    if medians is None:
        medians = {
            "distance_km": float(clean_df["distance_km"].median()),
            "parcel_weight_kg": float(clean_df["parcel_weight_kg"].median()),
            "declared_value_inr": float(clean_df["declared_value_inr"].median())
        }
    else:
        medians = {
            "distance_km": float(medians["distance_km"]),
            "parcel_weight_kg": float(medians["parcel_weight_kg"]),
            "declared_value_inr": float(medians["declared_value_inr"])
        }

    clean_df["distance_km"] = clean_df["distance_km"].fillna(medians["distance_km"])
    clean_df["parcel_weight_kg"] = clean_df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    clean_df["declared_value_inr"] = clean_df["declared_value_inr"].fillna(medians["declared_value_inr"])

    return clean_df, medians


def build_features(clean, hubs):
    clean_df = clean.merge(
        hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
        on="hub_id",
        how="left",
        validate="many_to_one"
    )

    clean_df["promised_hours"] = clean_df["service_type"].map(PROMISED_HOURS).astype(int)
    clean_df["is_cod"] = (clean_df["payment_mode"].str.upper() == "COD").astype(int)
    clean_df["is_weekend"] = (clean_df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    clean_df["dispatch_hour"] = clean_df["dispatch_time"].dt.hour.astype(int)
    clean_df["log_declared_value"] = np.log(clean_df["declared_value_inr"])

    return clean_df
