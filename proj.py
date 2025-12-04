# prepare_dataset.py

import pandas as pd
import math

# ---- Step 1: Load original Kaggle dataset ----
df = pd.read_csv("C:/Users/ashwi/Downloads/weightlifting_721_workouts.csv")

print("Columns after loading CSV:", df.columns.tolist()) # Added for debugging

# Expected columns in Kaggle dataset (may vary slightly)
# ['date', 'exercise', 'reps', 'weight']

# ---- Step 2: Add user personal info (edit this once) ----
df["gender"] = "M"
df["height_cm"] = 175
df["bodyweight_kg"] = 74

# ---- Step 3: Rename columns to match our project ----
df.rename(columns={
                   "Exercise Name": "workout_name", # Corrected from "exercise" to match actual CSV column
                   "Weight": "set_weight_kg",      # Corrected from "weight" to match actual CSV column
                   "Reps": "reps"                  # This was already correct
                   }, inplace=True)

# ---- Step 4: Add duration ----
# Assumption: each set ~3 mins; heavy sets ~4 mins
df["duration_min"] = df.apply(
    lambda r: 4 if r["reps"] < 6 else (3 if r["reps"] <= 12 else 2),
    axis=1
)

# ---- Step 5: MET based calories approximation ----
MET_LIFTING_HEAVY = 6.0     # barbell compound lifts
MET_LIFTING_LIGHT = 3.5     # machines / isolation
heavy_lifts = ["Bench Press", "Deadlift", "Squat", "Barbell Row", "Overhead Press"]

df["exercise_type"] = df["workout_name"].apply(
    lambda x: "heavy" if any(h in x for h in heavy_lifts) else "light"
)

df["met_value"] = df["exercise_type"].apply(
    lambda x: MET_LIFTING_HEAVY if x == "heavy" else MET_LIFTING_LIGHT
)

df["calories_burned"] = (df["met_value"] * df["bodyweight_kg"] * (df["duration_min"] / 60)).round(1)

# ---- Step 6: 1RM calculation ----
def calculate_1rm(weight, reps):
    if reps == 0 or weight == 0:
        return 0
    return weight * (1 + reps / 30)

df["predicted_1rm"] = df.apply(
    lambda r: calculate_1rm(r["set_weight_kg"], r["reps"]), axis=1
)

# ---- Step 7: Save final dataset ----
df_final = df[[
    "gender", "height_cm", "bodyweight_kg", "workout_name",
    "set_weight_kg", "reps", "duration_min",
    "calories_burned", "predicted_1rm"
]]

df_final.to_csv("workout_log.csv", index=False)
print("workout_log.csv created successfully!")