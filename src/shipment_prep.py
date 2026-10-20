import numpy as np
import pandas as pd


PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw, medians=None):
    #make a copy of the original data
    clean_df=raw.copy()
    #remove duplicate rows
    clean_df = clean_df.drop_duplicates()
    
    #clean service types
    clean_df["service_type"] = clean_df["service_type"].str.strip().str.title()
    
    #full missing payement mode with
    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("unknown")
    
    #convert declare values from text to number
    clean_df["declared_value"]= (clean_df["declared_value"].astype(str)
                                 .str.replace("₹", "", regex=False)
                                 .str.replace(",", "", regex=False).str.strip())
    clean_df["declared_value_inr"]= pd.to_numeric(clean_df["declared_value"], errors= "coerce")
    
    #remove the original text column
    clean_df= clean_df.drop(columns=["declared_value"])
    
    #convert dispatched time into date time
    clean_df["dispatch_time"]= pd.to_datetime(clean_df["dispatch_time"])
    
    # make distance and weight numeric
    clean_df["distance_km"]= pd.to_numeric(clean_df["distance_km"], errors="coerce")
    
    clean_df["parcel_weight_kg"]= pd.to_numeric(clean_df["parcel_weight_kg"], errors="coerce")
    
    
    #replace invalid scanner weight values with missing values >100
    clean_df.loc[clean_df["parcel_weight_kg"]>100, "parcel_weight_kg"]= np.nan
    
    #calculate medians usnning historical data
    if medians is None:
        medians={
            "distance_km": float(clean_df["distance_km"].median()),
            "parcel_weight_kg": float(clean_df["parcel_weight_kg"].median()),
            "declared_value_inr": float(clean_df["declared_value_inr"].median())
            
        }
        
    
    #fill missing numerical value using the medians
    clean_df["distance_km"]= clean_df["distance_km"].fillna(medians["distance_km"])
    clean_df["parcel_weight_kg"]= clean_df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    clean_df["declared_value_inr"]= clean_df["declared_value_inr"].fillna(medians["declared_value_inr"])
    
    #return the cleaned data and the medians
    return clean_df, medians


    
    
    