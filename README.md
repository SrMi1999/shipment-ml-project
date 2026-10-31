# Delhivery – Last-Mile Delivery Delay Prediction
## 1. Project Overview

  ## 2. Part A – Data Cleaning
  I started Part A by trying to understand the shipment data before moving on to machine learning. My first goal was to make the data consistent, create the features required by the assignment, and check whether the shipment information showed any patterns related to delivery time and delays.

  I used the assignment's data generator and treated the generated records as the project dataset. The results in this README describe this dataset only; they are not actual Delhivery operational statistics.

   ### 2.1 What I Did
   I loaded the shipment data and the hub details from the data folder. Before filling or changing values, I checked the row count, exact duplicates, missing values, inconsistent service-type spellings, missing payment modes, and parcel weights that looked invalid.

   I removed exact duplicate rows first. The raw file had 1,224 rows, and 24 of them were exact duplicates, leaving 1,200 shipment rows. I used the deduplicated data for the cleaning checks so that repeated rows would not inflate the counts.

   Next, I cleaned service_type by stripping extra spaces and converting the names to title case. This reduced the number of distinct spellings from 12 to 3. I filled missing payment_mode values with the exact placeholder Unknown rather than guessing which payment method was used.

   The declared_value field was stored as text and included rupee symbols and comma separators. I removed those characters, converted the values to numbers, and stored the result in declared_value_inr. I converted dispatch_time to datetime as well, so I could derive the dispatch hour and weekend flag from it.

   The assignment identifies parcel weights above 100 kg as scanner-fault values. I treated those 5 values as missing instead of allowing them to affect the weight statistics. I then filled missing distance_km, parcel_weight_kg, and declared_value_inr values with their medians. The medians used were 314.95 km, 1.83 kg, and INR 1,121.50. I returned these median values in a dictionary because the same cleaning logic may need to be applied to new shipments later.

   After cleaning, I joined the hub information to the shipment data with a left merge on hub_id. The merge kept all 1,200 shipment rows. I created the features listed in the assignment, including promised_hours, is_cod, is_weekend, dispatch_hour, and log_declared_value.

   Finally, I created the target is_delayed. A shipment is marked as delayed- 1 when delivery_hours is greater than promised_hours; otherwise, it is marked as not delayed (0). I kept delivery_hours as the outcome used for analysis, not as one of the 17 model features.

   You can see the out put from outputs/part_a_checks.csv.

   The checks helped me confirm that the row count stayed consistent after the hub merge and that the required feature and outcome columns were complete after cleaning. The missing-value counts above are taken before imputation, after duplicate removal. The five invalid weights are counted separately from the weights that were already missing.

   ### 2.2 What I Observed from the Delay Results
   
   There were 278 delayed shipments out of 1,200, which gives an overall delay rate of 23.17% in this generated dataset. I then compared delay rates by service type, hub tier, and distance band instead of relying only on the overall rate.

   Express had the highest delay rate: 119 out of 243 Express shipments were delayed. Economy had the lowest rate at 8.57%, while Standard was at 20.56%. I found this worth checking because Express has a shorter promised delivery time of 36 hours, so a shipment can be classified as delayed even when its absolute delivery time is lower than the mean for Economy. The delay label depends on the service promise, not only on the number of delivery hours.

   The delay rate increased from 4.00% for Tier 1 hubs to 21.50% for Tier 2 and 44.00% for Tier 3. In this dataset, Tier 3 had eleven times the delay rate of Tier 1. This is a strong pattern to investigate in later parts of the project, but this comparison alone does not prove that hub tier causes delays.

   The delay rate increased across the distance bands. It was 2.19% for shipments up to 200 km and 85.23% for the 601+ km group. The last group contains only 88 shipments, so I would keep its smaller sample size in mind when interpreting the rate. I used the distance bands specified in the assignment so that the group results are consistent and reproducible.

   ### 2.3 Correlation Findings

   I generated the full numeric correlation matrix and saved it to outputs/part_a_correlations.csv. To write this finding, I ranked the correlations with delivery_hours by their absolute values and excluded delivery_hours itself from the ranking.
   
   promised_hours had the strongest positive correlation with delivery duration, followed by distance and the number of stops. The direction of these correlations means that higher values tended to occur alongside higher delivery hours in this dataset. Correlation shows association, not cause and effect, so I would not use these numbers alone to conclude that changing one feature would directly change delivery time.

   ### 2.4 Checks and Improvements I Made While Preparing Part A

   While reviewing the pipeline, I focused on making the checks reflect what the script actually does instead of only printing values that looked reasonable. I made sure that duplicate rows were removed before the “before filling” missing-value counts were calculated. I also kept invalid weights separate from values that were already missing, so those two checks would not be mixed together.

   I checked that missing payment modes use the required Unknown spelling, that the delay target is represented as integer values (0 and 1), and that the null check includes all 17 features, delivery_hours, and is_delayed. I also made the delay evidence output include both counts and mean_hours, and used the assignment's required distance bands. For correlations, I saved the complete numeric matrix rather than only a few selected feature pairs. These checks make the outputs easier to verify and reuse in the next parts of the project.


   ### 2.5 Current Conclusion
  
   Part A gives me a clean modelling table and an initial picture of where delays are more common in the generated data. The main patterns I found are the higher delay rate for Express shipments, the increasing delay rate across hub tiers, and the strong increase in delay rate for longer-distance shipments. These are useful findings to carry into the next parts, where the focus can move from describing the data to building and evaluating a model.



   ### 2.6 How to Run Part A

   From the project root, after setting up the Python environment from requirements.txt, run:

   python generate_data.py
   python part_a_clean_eda.py





  ## 3. Part B – Model Results (Regression)


   ### 3.1 What I Did

   For Part B, I used the modelling table created in Part A. I selected the 17 required features in the same order as the FEATURES list and used delivery_hours as the target. I split the 1,200 shipments into 840 training rows, 180 validation rows and 180 test rows using the specified random state.

   I fitted the diagnostic StandardScaler on the training rows only and then transformed the test rows using the same fitted scaler. The maximum absolute mean of the scaled training features was approximately 0.0000, with an actual value of 4.0391e-16. The corresponding value for the test features was 0.2187. This was consistent with fitting the scaler on training data rather than fitting it separately on the test data.

   ### 3.2 Baseline and Model Comparison
   I first created a mean baseline that predicts the mean delivery time from the training data for every validation shipment. Its validation R² was -0.0212. This means that the baseline did not explain the variation in validation delivery times effectively; its predictions were slightly worse than using the validation target mean as a reference.

   I then compared Linear Regression, Decision Tree, Random Forest and Gradient Boosting on the same validation split. Linear Regression had the lowest validation RMSE at 4.8912 hours, with an R² of 0.9313. Gradient Boosting was close behind, with a validation RMSE of 4.9165 hours and an R² of 0.9306. Random Forest achieved an R² of 0.9128, while the selected Decision Tree achieved 0.8867.

   Based on the required validation-RMSE selection rule, I selected Linear Regression as the final model. When evaluated once on the reserved test data, it achieved an RMSE of 4.5235 hours and an R² of 0.9493.

   ### 3.3 What I Observed from Tree Depth
   The depth experiment showed how changing the complexity of a Decision Tree affected its errors. At depth 1, the training RMSE was 11.6334 and the validation RMSE was 12.1324. Both errors were high, and their difference was only 0.4990. This suggested high bias because the shallow tree was too simple to capture the patterns in the data.

   Validation RMSE decreased as the depth increased up to depth 6. At depth 6, it reached its lowest value of 6.2796 hours. After that, validation RMSE increased even though training RMSE continued to fall. At depth 12, the training and validation RMSE values were 1.3613 and 7.3540, respectively. With unlimited depth, the training RMSE reached 0.0000, but the validation RMSE increased to 7.8368. The gap of 7.8368 hours indicated high variance and strong overfitting.
   For this split, depth 6 provided the best validation result among the tested tree configurations.

   ### 3.4 What I Observed from the Learning-Rate Experiment
   I tested Gradient Boosting with learning rates of 0.01, 0.1 and 1.0, keeping the number of estimators at 100. A learning rate of 0.01 performed poorly, with a validation R² of 0.7127 and RMSE of 10.0018 hours. With only 100 estimators, the smaller updates appeared too slow to learn the pattern sufficiently.

   The learning rate of 0.1 gave the best validation result, with an R² of 0.9306 and RMSE of 4.9165 hours. At 1.0, the R² fell to 0.8440 and RMSE increased to 7.3701 hours. This suggests that the larger updates overshot a better fit. Among these three settings, 0.1 was the best choice.

   ### 3.5 Cross-Validation Comparison and Final Decision
   I selected the two best candidates by validation RMSE and evaluated them using the same five-fold KFold configuration on the combined training and validation data. Linear Regression achieved a mean cross-validation R² of 0.9261 with a standard deviation of 0.0048. Gradient Boosting achieved a mean R² of 0.9353 with a standard deviation of 0.0045.

   The cross-validation ranking therefore favoured Gradient Boosting, while the single validation split favoured Linear Regression. The difference is small, and the cross-validation results show that Gradient Boosting also performed consistently across the folds. However, the Part B instructions specify that the final model must be the candidate with the lowest validation RMSE. I therefore kept Linear Regression as the selected model and reported its test results separately rather than changing the selection rule after looking at the cross-validation scores.



  ## 4. Part C – Classification: Predicting Delayed Shipments

   ### 4.1 What I Wanted to Find Out
   In Part B, I predicted the delivery time of a shipment in hours. In Part C, I wanted to solve a different problem: whether a shipment would be delayed or not.

   For this part, I treated is_delayed as the target variable:0 means the shipment was delivered within the promised time.
   1 means the shipment took longer than the promised time

   I have to compare different classification models, understand which features were useful, check whether the bookkeeping columns affected the model, and choose a probability threshold that could identify delayed shipments without missing too many of them.  

   ### 4.2 Train, Validation and Test Split
   I used a stratified split so that the proportion of delayed and non-delayed shipments was maintained as closely as possible in each dataset.
   I used the training data to fit the models, the validation data to compare them and select the threshold, and the test data for the final evaluation.
   I kept the test set separate during model selection so that I could use it to check how the final model performed on unseen data.

   ### 4.3 Comparing the Classification Models
   I started by comparing the required models on the validation set. I also included a simple baseline that predicted every shipment as not delayed.
   The blind baseline achieved 76.67% accuracy. At first, this looked reasonably high, but it predicted every shipment as on time. It failed to identify all 42 delayed shipments in the validation set.

   This showed me why accuracy alone can be misleading when the objective is to identify delayed shipments. A model can achieve high accuracy while failing to detect the class that matters most.

   Logistic Regression achieved an F1-score of 0.8046, while the Decision Tree with depth 6 achieved 0.7470. The default Random Forest achieved 0.7848.

   Based on these results, I could see that the models were learning useful patterns, but I still needed to investigate the features and test whether selecting fewer features could improve the Random Forest.

   ### 4.4 Feature Importance and the Bookkeeping Columns
   
   I calculated the F-scores using SelectKBest(f_classif) and calculated feature importance using a Random Forest trained on the training data.

   Distance was the strongest individual feature according to the F-score. Hub tier, hub capacity and the number of stops also showed strong relationships with the delay target in this dataset.

   These scores measure how strongly an individual feature is associated with the target. They do not prove that the feature causes a delay, and they do not measure the contribution of the complete feature set.

   I also checked the five bookkeeping columns: label_print_batch, scanner_build_no, tape_roll_no, sticker_serial_mod and shelf_slot_no.
   All five ranked outside the top seven by F-score, and each had an F-score below 10. However, their combined Random Forest importance was 0.2244, or 22.44%.

   his was an important observation. Although these columns did not rank among the strongest individual features by F-score, the Random Forest still assigned them some combined importance. I therefore did not assume that their influence was zero just because their F-scores were low.

   ### 4.5 Selecting the Number of Features
   Next, I compared Random Forest models using different values of k with SelectKBest.
   The model using five features performed worse than the models using seven or ten features. The displayed results for seven and ten features were both 0.8140.

   Since the required final model uses seven selected features, I proceeded with k=7. This also allowed me to compare the tuned models and the default Random Forest using the same selected feature set.

   ### 4.6 Grid Search, Randomized Search and the Default Random Forest
   I compared three candidates using the seven selected features.For Grid Search, I tested 27 parameter combinations with five-fold cross-validation, resulting in 135 fits. For Randomized Search, I tested nine candidates with five-fold cross-validation, resulting in 45 fits.Grid Search achieved a slightly higher cross-validation accuracy than Randomized Search. However, its validation F1-score was lower.


   This difference helped me understand that the best cross-validation accuracy does not necessarily mean the highest validation F1-score. The searches used the default classifier scoring of accuracy, whereas I used F1-score to compare the candidates on the separate validation set.

   The default Random Forest achieved the highest validation F1-score of 0.8140. Therefore, I selected default_forest_same_7_features as the final candidate.

   The validation set contained only 180 shipments, so I treated the differences between these scores cautiously. They were useful for model selection, but they were not enough to prove that one configuration would always perform better on other data.

   ### 4.7 Choosing the Alert Threshold
   The model produces a probability for each shipment. I tested thresholds from 0.1 to 0.9, classifying a shipment as delayed when its predicted probability was greater than or equal to the selected threshold.
   My main operational requirement was to maintain validation recall of at least 0.80. I therefore selected the highest tested threshold that satisfied that condition.

   At threshold 0.4, the model detected more delayed shipments, but it also generated more false alerts. At threshold 0.6, precision increased, but recall fell below the required 0.80.

   Threshold 0.5 was the highest tested threshold that maintained the required validation recall. I therefore selected 0.5 as the final alert threshold, even though the displayed F1-score at 0.6 was slightly higher.

   The model identified 35 of the 42 delayed validation shipments and missed seven. It also flagged nine shipments that were not actually delayed.
   For this project, I preferred maintaining the required recall rather than choosing a higher threshold that would miss more delayed shipments.

   ### 4.8 Final Test Evaluation
   After selecting the model and threshold using the training and validation process, I evaluated the final model on the held-out test set.

  #### The final configuration was:  Feature selection: SelectKBest with k=7
        Model: Random Forest, 
        Number of trees: 100, 
        Maximum depth: Unlimited,
        Minimum samples per leaf: 1,
        Alert threshold: 0.5

  The final model correctly identified 31 of the 41 delayed test shipments, while missing ten. It also generated 14 false alerts.
  The test F1-score was 0.7209, which was lower than the validation F1-score of 0.8140. Precision and recall also decreased on the test set.
  I did not treat this difference as proof that the model had failed. The test set was separate from the validation set, and its results provide a more realistic check of how the selected configuration performed on unseen shipments. However, the difference does show that validation performance should not be treated as a guarantee of test performance.

  The test set contained only 41 delayed shipments, so the recall estimate is sensitive to a small number of predictions. For example, each additional delayed shipment correctly identified or missed changes test recall by approximately 2.44 percentage points.


  ### 4.9 What I Learned from Part C

  This part helped me understand that classification is not only about getting a high accuracy score. The blind baseline had 76.67% accuracy but did not identify any delayed shipments. Feature selection helped me compare smaller feature sets, and the tuning experiments showed that higher cross-validation accuracy does not automatically produce a higher validation F1-score.

  I also learned that the threshold should be selected according to the practical requirement. In this case, maintaining recall of at least 0.80 on the validation set led me to choose the threshold of 0.5 instead of 0.6.

  Finally, the lower test F1-score reminded me that model performance can change on unseen data. A useful next step would be to evaluate the model on a larger set of real shipment records and monitor its precision, recall and false-alert rate over time.

  The detailed classification results are saved in the following files:

  outputs/part_c_models.csv,
  outputs/part_c_feature_scores.csv,
  outputs/part_c_selectk.csv,
  outputs/part_c_tuning.csv,
  outputs/part_c_thresholds.csv,
  outputs/part_c_final_test.csv,
  outputs/part_c_final_config.csv,



## 5. Part D – Hub Segmentation and Prediction
 ### 5.1 Hub Segmentation
 
 In Part D, I grouped shipments by hub_id and calculated the shipment count, average delivery hours, delay rate, average distance, and average number of stops.
 The resulting hub profile contained 24 hubs, with 50 shipments per hub.
 I scaled the four hub-level measurements and applied KMeans clustering with random_state=42 and n_init=10, I compared K values from 2 to 5 using the across-cluster distance divided by within-cluster distanc. Although K=4 and K=5 had higher separation ratios, they created smaller clusters. I selected K=3 as a more practical grouping for hub operations.
 I named the clusters using their average delay rates: Standard, Watch, Escalate as given

 ### 5.2 Hub Tier Mismatches

 I compared the KMeans segments with the expected tier mapping: Tier 1 – Standard, Tier 2 – Watch, and Tier 3 – Escalate.
 I found four mismatches
 These hubs are worth reviewing because their observed shipment patterns do not match the expected tier grouping.


 ### 5.3 Model Packaging
 I created package_model.py to rebuild the final Part C model using the selected configuration.

 #### I saved the trained model and its supporting files:
  models/delay_model.pkl,
  models/delay_threshold.csv,
  models/feature_order.csv,
  models/impute_values.csv,

  I then loaded the saved model again and compared its probabilities with the original model's probabilities on the test data.
  The maximum probability difference after reloading was 0.0, confirming that the saved model reproduced the original predictions in this run.

  ### 5.4 Prediction on 48 New Shipments
  Finally, I used predict_delay.py to process the 48 new shipments provided by the data generator.
  The script reused the cleaning and feature-building functions, loaded the saved model and threshold, and joined the hub segment information.
  A total of 48 new shipments were processed during the reporting period. Out of these, 40 shipments required no further action, as they proceeded without issue.
  However, the remaining items required specific operational interventions. A total of 8 shipments were flagged for review. This resulted in 6 escalations to the hub manager for resolution and prompted 2 proactive customer calls to address potential concerns.

  ### 5.5 The action rules were:


  Flagged + Escalate segment: Escalate to hub manager

  Flagged + Standard or Watch segment: Proactive customer call

  Not flagged: No action

  I checked that all 48 shipment IDs were unique and that the output had no missing values. I also ran the prediction script twice. Both runs produced the same SHA256 hash:

  879EE4FBC04CEA318DB11BF9C1A1D447D9F6484ABF8248A3297CFFAAD835D7B6

  This confirmed that the prediction output was reproducible for the same input files and environment.


  The model was packaged using the scikit-learn version pinned in requirements.txt. If the local scikit-learn version is different, I recommend running package_model.py again before running predictions.

## 6. Production PlanProduction Plan & Limitations-
  ### 6.1 Poduction Plan


  Before using this workflow in a real delivery operation, I would connect it to regularly updated shipment and hub data and validate it on real historical records.

  ### I would focus on these areas:

   #### Accuracy: 
   Track test and future production precision, recall, F1-score, RMSE, and R².
   #### Reliability: 
   Keep the data checks and prediction validation used in Parts A and D.
   #### Reproducibility:
   Keep fixed random states, pinned package versions, and repeatable scripts.
   #### Maintainability:
   Reuse the shared cleaning and feature-building functions in src/shipment_prep.py.
   #### Consistency: 
   Load the saved feature order, imputation values, model, and threshold together.
   Monitorability: Track hub delay rates, segment changes, flagged shipment rates, and prediction outcomes.

  I would initially monitor these operational indicators weekly. The following are proposed warning triggers, not results observed in this project:

  Overall delay rate above 28%.
  Flagged shipment share above 25%.
  Recall on shipments with confirmed delivery outcomes below 75%.
  Missing required model inputs above 0%.
  I would review these indicators with operations teams and investigate unusual changes before changing the model.
  I would also review model performance every three months and retrain earlier if there is sustained data or performance drift.

  The assignment's generator includes a sale-event period from September 15 to September 28. I would monitor this period separately and compare it with previous sale-event periods rather than comparing it only with ordinary weeks. This helps distinguish expected seasonal changes from unusual operational problems.

  Data drift means that the incoming shipment data changes, while concept drift means that the relationship between shipment features and delivery delays changes. Both should be checked before deciding to retrain.

## 6.2. Limitations

The dataset is generated for the assignment and does not represent actual Delhivery operations.

The classification test set contains only 41 delayed shipments, so a small number of prediction changes can affect the reported metrics.

The classification threshold was selected using one validation split. It should be checked again using more historical data.

The hub profiles contain 50 shipments per hub, which is a limited sample for making long-term operational decisions.

The dataset may not include every real-world factor affecting delivery, such as unexpected route closures, staffing shortages, or local disruptions.

Overall, this project helped me build a complete workflow from data cleaning and model training to hub-level analysis and actionable predictions for new shipments.







## 7. How to Run

used Python with the package versions listed in requirements.txt.

 ### From the project folder, create and activate a virtual environment:

   python -m venv .venv
  .\.venv\Scripts\Activate.ps1

  Install the required packages:

  pip install -r requirements.txt

  Run the scripts in this order:

  python generate_data.py
  python part_a_clean_eda.py
  python part_b_regression.py
  python part_c_classification.py
  python part_d_segments.py
  python package_model.py
  python predict_delay.py

The scripts generate the datasets, analysis outputs, hub segments, saved model files, and new-shipment action report.





