import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("C:/Users/ashwi/OneDrive/Documents/python/workout_log.csv")

X = df[["gender", "height_cm", "bodyweight_kg", "workout_name",
        "set_weight_kg", "reps", "duration_min"]]
y = df["calories_burned"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["gender", "workout_name"]),
        ("num", StandardScaler(), ["height_cm", "bodyweight_kg", "set_weight_kg", "reps", "duration_min"])
    ]
)

model = Pipeline(steps=[("preprocessor", preprocessor),
                       ("regressor", RandomForestRegressor(n_estimators=250))])

model.fit(X, y)
joblib.dump(model, "calories_model.joblib")
print("Model saved as calories_model.joblib")
