Delhivery - Capstone project-

   Part A- data cleaning-
   I started Part A by trying to understand the shipment data before moving on to machine learning. My first goal was to make the data consistent, create the features required by the assignment, and check whether the shipment information showed any patterns related to delivery time and delays.

   I used the assignment's data generator and treated the generated records as the project dataset. The results in this README describe this dataset only; they are not actual Delhivery operational statistics.

   What I did-
   I loaded the shipment data and the hub details from the data folder. Before filling or changing values, I checked the row count, exact duplicates, missing values, inconsistent service-type spellings, missing payment modes, and parcel weights that looked invalid.

   I removed exact duplicate rows first. The raw file had 1,224 rows, and 24 of them were exact duplicates, leaving 1,200 shipment rows. I used the deduplicated data for the cleaning checks so that repeated rows would not inflate the counts.

   Next, I cleaned service_type by stripping extra spaces and converting the names to title case. This reduced the number of distinct spellings from 12 to 3. I filled missing payment_mode values with the exact placeholder Unknown rather than guessing which payment method was used.

   The declared_value field was stored as text and included rupee symbols and comma separators. I removed those characters, converted the values to numbers, and stored the result in declared_value_inr. I converted dispatch_time to datetime as well, so I could derive the dispatch hour and weekend flag from it.

   The assignment identifies parcel weights above 100 kg as scanner-fault values. I treated those 5 values as missing instead of allowing them to affect the weight statistics. I then filled missing distance_km, parcel_weight_kg, and declared_value_inr values with their medians. The medians used were 314.95 km, 1.83 kg, and INR 1,121.50. I returned these median values in a dictionary because the same cleaning logic may need to be applied to new shipments later.

   After cleaning, I joined the hub information to the shipment data with a left merge on hub_id. The merge kept all 1,200 shipment rows. I created the features listed in the assignment, including promised_hours, is_cod, is_weekend, dispatch_hour, and log_declared_value.

   Finally, I created the target is_delayed. A shipment is marked as delayed- 1 when delivery_hours is greater than promised_hours; otherwise, it is marked as not delayed (0). I kept delivery_hours as the outcome used for analysis, not as one of the 17 model features.

   You can see the out put from outputs/part_a_checks.csv.

   The checks helped me confirm that the row count stayed consistent after the hub merge and that the required feature and outcome columns were complete after cleaning. The missing-value counts above are taken before imputation, after duplicate removal. The five invalid weights are counted separately from the weights that were already missing.

   What I observed from the delay results-
   There were 278 delayed shipments out of 1,200, which gives an overall delay rate of 23.17% in this generated dataset. I then compared delay rates by service type, hub tier, and distance band instead of relying only on the overall rate.

   Express had the highest delay rate: 119 out of 243 Express shipments were delayed. Economy had the lowest rate at 8.57%, while Standard was at 20.56%. I found this worth checking because Express has a shorter promised delivery time of 36 hours, so a shipment can be classified as delayed even when its absolute delivery time is lower than the mean for Economy. The delay label depends on the service promise, not only on the number of delivery hours.

   The delay rate increased from 4.00% for Tier 1 hubs to 21.50% for Tier 2 and 44.00% for Tier 3. In this dataset, Tier 3 had eleven times the delay rate of Tier 1. This is a strong pattern to investigate in later parts of the project, but this comparison alone does not prove that hub tier causes delays.

   The delay rate increased across the distance bands. It was 2.19% for shipments up to 200 km and 85.23% for the 601+ km group. The last group contains only 88 shipments, so I would keep its smaller sample size in mind when interpreting the rate. I used the distance bands specified in the assignment so that the group results are consistent and reproducible.

   Correlation findings-
    I generated the full numeric correlation matrix and saved it to outputs/part_a_correlations.csv. To write this finding, I ranked the correlations with delivery_hours by their absolute values and excluded delivery_hours itself from the ranking.
   The three strongest correlations were:
    Feature	Correlation  with delivery_hours
    promised_hours	-     +0.8372
    distance_km 	-     +0.4013
    n_stops 	     -    +0.3247


    promised_hours had the strongest positive correlation with delivery duration, followed by distance and the number of stops. The direction of these correlations means that higher values tended to occur alongside higher delivery hours in this dataset. Correlation shows association, not cause and effect, so I would not use these numbers alone to conclude that changing one feature would directly change delivery time.

    Checks and improvements I made while preparing Part A-

    While reviewing the pipeline, I focused on making the checks reflect what the script actually does instead of only printing values that looked reasonable. I made sure that duplicate rows were removed before the “before filling” missing-value counts were calculated. I also kept invalid weights separate from values that were already missing, so those two checks would not be mixed together.

    I checked that missing payment modes use the required Unknown spelling, that the delay target is represented as integer values (0 and 1), and that the null check includes all 17 features, delivery_hours, and is_delayed. I also made the delay evidence output include both counts and mean_hours, and used the assignment's required distance bands. For correlations, I saved the complete numeric matrix rather than only a few selected feature pairs. These checks make the outputs easier to verify and reuse in the next parts of the project.


    Current conclusion
    Part A gives me a clean modelling table and an initial picture of where delays are more common in the generated data. The main patterns I found are the higher delay rate for Express shipments, the increasing delay rate across hub tiers, and the strong increase in delay rate for longer-distance shipments. These are useful findings to carry into the next parts, where the focus can move from describing the data to building and evaluating a model.

How to run Part A-
From the project root, after setting up the Python environment from requirements.txt, run:

python generate_data.py
python part_a_clean_eda.py



Part_B_Model results (regression)-

What I did - For Part B, I used the modelling table created in Part A. I selected the 17 required features in the same order as the FEATURES list and used delivery_hours as the target. I split the 1,200 shipments into 840 training rows, 180 validation rows and 180 test rows using the specified random state.

I fitted the diagnostic StandardScaler on the training rows only and then transformed the test rows using the same fitted scaler. The maximum absolute mean of the scaled training features was approximately 0.0000, with an actual value of 4.0391e-16. The corresponding value for the test features was 0.2187. This was consistent with fitting the scaler on training data rather than fitting it separately on the test data.

Baseline and model comparison- I first created a mean baseline that predicts the mean delivery time from the training data for every validation shipment. Its validation R² was -0.0212. This means that the baseline did not explain the variation in validation delivery times effectively; its predictions were slightly worse than using the validation target mean as a reference.

I then compared Linear Regression, Decision Tree, Random Forest and Gradient Boosting on the same validation split. Linear Regression had the lowest validation RMSE at 4.8912 hours, with an R² of 0.9313. Gradient Boosting was close behind, with a validation RMSE of 4.9165 hours and an R² of 0.9306. Random Forest achieved an R² of 0.9128, while the selected Decision Tree achieved 0.8867.

Based on the required validation-RMSE selection rule, I selected Linear Regression as the final model. When evaluated once on the reserved test data, it achieved an RMSE of 4.5235 hours and an R² of 0.9493.

What I observed from tree depth- The depth experiment showed how changing the complexity of a Decision Tree affected its errors. At depth 1, the training RMSE was 11.6334 and the validation RMSE was 12.1324. Both errors were high, and their difference was only 0.4990. This suggested high bias because the shallow tree was too simple to capture the patterns in the data.

Validation RMSE decreased as the depth increased up to depth 6. At depth 6, it reached its lowest value of 6.2796 hours. After that, validation RMSE increased even though training RMSE continued to fall. At depth 12, the training and validation RMSE values were 1.3613 and 7.3540, respectively. With unlimited depth, the training RMSE reached 0.0000, but the validation RMSE increased to 7.8368. The gap of 7.8368 hours indicated high variance and strong overfitting.
For this split, depth 6 provided the best validation result among the tested tree configurations.

What I observed from the learning-rate experiment- I tested Gradient Boosting with learning rates of 0.01, 0.1 and 1.0, keeping the number of estimators at 100. A learning rate of 0.01 performed poorly, with a validation R² of 0.7127 and RMSE of 10.0018 hours. With only 100 estimators, the smaller updates appeared too slow to learn the pattern sufficiently.

The learning rate of 0.1 gave the best validation result, with an R² of 0.9306 and RMSE of 4.9165 hours. At 1.0, the R² fell to 0.8440 and RMSE increased to 7.3701 hours. This suggests that the larger updates overshot a better fit. Among these three settings, 0.1 was the best choice.

Cross-validation comparison and final decision- I selected the two best candidates by validation RMSE and evaluated them using the same five-fold KFold configuration on the combined training and validation data. Linear Regression achieved a mean cross-validation R² of 0.9261 with a standard deviation of 0.0048. Gradient Boosting achieved a mean R² of 0.9353 with a standard deviation of 0.0045.

The cross-validation ranking therefore favoured Gradient Boosting, while the single validation split favoured Linear Regression. The difference is small, and the cross-validation results show that Gradient Boosting also performed consistently across the folds. However, the Part B instructions specify that the final model must be the candidate with the lowest validation RMSE. I therefore kept Linear Regression as the selected model and reported its test results separately rather than changing the selection rule after looking at the cross-validation scores.





