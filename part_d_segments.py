import os #Used to create the output folder
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)#create the out put folder if it doesn't exist
#read the modeling table created in part a 
modelling_table = pd.read_csv("data/modelling_table.csv")

#calculate one operating profile for every hub
hub_profile= (modelling_table.groupby("hub_id", as_index=False).agg(shipments=("hub_id", "size"),mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    ).sort_values("hub_id").reset_index(drop=True)
)

#save the hub profile to a csv file- hub_profile.csv
hub_profile.round(4).to_csv("outputs/hub_profile.csv",index=False)

#K means segmantation
#selcet the 4 operating measurements for clustering
profile_features = ["mean_delivery_hours","delay_rate","mean_distance_km","mean_n_stops"]

#sacle the measurements so they are on a comparable scale
scaler = StandardScaler()
scaled_profile = scaler.fit_transform(hub_profile[profile_features])
#calculate the distance betwwen every pair of hubs
pair_distances = (np.linalg.norm(scaled_profile[:, np.newaxis, :]- scaled_profile[np.newaxis, :, :], axis=2))

#keep only unordered pairs of hubs so that each pair is only counted once
upper_row, upper_col = np.triu_indices(len(hub_profile), k=1)
all_pair_distances = pair_distances[upper_row, upper_col]

#check cluster separation for K = 2,3,4 and 5
separation_rows = []

for k in range(2,6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_profile)
    #separate the pairs within the same cluster and across different clusters
    same_cluster = (cluster_labels[upper_row] == cluster_labels[upper_col])
    different_cluster = ~same_cluster
    within_mean = all_pair_distances[same_cluster].mean()
    across_mean = all_pair_distances[different_cluster].mean()
    across_over_within = across_mean / within_mean
    
    #count the number of hubs in each cluster, from largest to smallest
    cluster_sizes = sorted(pd.Series(cluster_labels).value_counts().tolist(),reverse=True)
    separation_rows.append({"K": k, "across_over_within": across_over_within, "cluster_sizes": "-".join(str(size) for size in cluster_sizes)})
    
#save the separation result to part_d_separation.csv
pd.DataFrame(separation_rows, columns=["K", "across_over_within", "cluster_sizes"]).round(4).to_csv("outputs/part_d_separation.csv",index=False)

#K = 3 is fixed by the business — operations runs exactly three playbooks.
final_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(scaled_profile)

#calculate the avarage delay rate of each cluster
cluster_delay_rates = (hub_profile.assign(cluster=final_labels).groupby("cluster")["delay_rate"].mean().sort_values())

#Segment names and tier check.
segment_names = ["Standard", "Watch", "Escalate"]
cluster_to_segment = {cluster: segment_names[position]
    for position, cluster in enumerate(cluster_delay_rates.index)}

#create the hub segmentation table
hub_segments = hub_profile[["hub_id"]].copy()
hub_segments["segment"] = [cluster_to_segment[label] for label in final_labels]

#adding the city and tier information hubs.csv
hubs = pd.read_csv("data/hubs.csv")
hub_segments = hub_segments.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left", validate="one_to_one").sort_values("hub_id")

#keep the require column order
hub_segments = hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]]

#save the final hub segments to a csv file
hub_segments.to_csv("outputs/hub_segments.csv",index=False)

#calculate the average operating profile of each segments
segment_order = ["Standard", "Watch", "Escalate"]
segment_profile = (hub_profile.assign(segment=hub_segments["segment"].values).groupby("segment", as_index=False).agg(hubs=("hub_id", "size"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")))


#keep the segments in the required business order
segment_profile["segment_order"] = segment_profile["segment"].map(
    {
        name: position
        for position, name in enumerate(segment_order)
    }
)

segment_profile = (segment_profile.sort_values("segment_order").drop(columns="segment_order").reset_index(drop=True))

#save the segment profile to a csv file
segment_profile.round(4).to_csv("outputs/segment_profile.csv",index=False)

#check weather the actual segments matches the hub tier
tier_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

expected_segment = hub_segments["hub_tier"].map(tier_segment)
mismatches = hub_segments.loc[hub_segments["segment"] != expected_segment].copy()

#save the hubs whose tier and segment do not match
mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

#print a short summary of the results
print("\n========== Part D Hub Segmentation Summary ==========")
print("Hubs profiled:", len(hub_profile))
print("Shipments per hub:", int(hub_profile["shipments"].min()), "to",int(hub_profile["shipments"].max()))
parts = []
for row in segment_profile.itertuples():
    parts.append(f"{row.segment}: {int(row.hubs)}")

output = ", ".join(parts)
print(f"Segment sizes: {output}")

print("Tier-versus-segment mismatches:", len(mismatches))
print("Part D hub segmentation completed.", "Five CSV output files saved in outputs.")








    

