import pandas as pd
import joblib
import os
INCOMING_FILE = "incoming_credit.csv"


# Read the new data
incoming_raw = pd.read_csv(INCOMING_FILE) 

# Load what we saved in the training script
model = joblib.load("artifacts/model.pkl") 
training_cols = joblib.load("artifacts/feature_columns.pkl") 

# PREPARATION STEPS (One step at a time)
# Step 1: Create dummy variables
encoded_data = pd.get_dummies(incoming_raw, drop_first=True)

# Step 2: Ensure columns match the training format exactly
final_features = encoded_data.reindex(columns=training_cols, fill_value=0)

# PREDICT
predictions = model.predict(final_features) 
predictions_probability = model.predict_proba(final_features)[:,1]

# SAVE RESULTS 
scored_df = incoming_raw.copy()
scored_df["pred_default"] = predictions 
scored_df["pred_probability"] = predictions_probability

# Save the file as csv
scored_df.to_csv("incoming_scored.csv", index=False)
print("Success! 'incoming_scored.csv' has been created.")
