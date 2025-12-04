import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Try to load ML model (optional)
model = None
try:
    import joblib
    model = joblib.load("calories_model.joblib")
    model_status = "ML model loaded (calories_model.joblib)."
except Exception:
    model_status = "No ML model found. Using simple formula for calories."

HISTORY_FILE = "workout_history.csv"

# -----------------------------
# Helper functions
# -----------------------------
def estimate_1rm_rpe(weight_kg: float, reps: int, rpe: float) -> float:
    """
    Estimate 1RM using weight, reps, and RPE.

    RPE 10 -> 0 RIR
    RPE 9  -> 1 RIR
    RPE 8  -> 2 RIR
    ...
    We then estimate reps_at_failure = reps + RIR
    and apply Epley formula on reps_at_failure.
    """
    if weight_kg <= 0 or reps <= 0 or rpe < 1 or rpe > 10:
        return 0.0

    rir = 10 - rpe  # e.g., RPE 8 -> 2 RIR
    rir = max(0, min(5, rir))  # clamp between 0 and 5

    reps_at_failure = reps + rir
    return weight_kg * (1 + reps_at_failure / 30.0)

def get_met_value(workout_name: str) -> float:
    """
    Simple mapping from workout name to MET value.
    You can tweak this mapping.
    """
    name = workout_name.lower()
    heavy_lifts = ["bench", "deadlift", "squat", "row", "press", "clean"]
    if any(h in name for h in heavy_lifts):
        return 6.0   # heavy strength training
    else:
        return 3.5   # general/light strength work

def formula_calories(bodyweight_kg: float, workout_name: str, duration_min: float) -> float:
    """
    Estimate calories using MET formula:
    kcal = MET * weight_kg * (duration_hr)
    """
    if bodyweight_kg <= 0 or duration_min <= 0:
        return 0.0
    met = get_met_value(workout_name)
    duration_hr = duration_min / 60.0
    return met * bodyweight_kg * duration_hr

def predict_calories(
    gender: str,
    height_cm: float,
    bodyweight_kg: float,
    workout_name: str,
    set_weight_kg: float,
    reps: int,
    duration_min: float,
) -> float:
    """
    If ML model is available, use it.
    Otherwise, use formula-based estimate.
    """
    global model
    if model is not None:
        input_df = pd.DataFrame(
            {
                "gender": [gender],
                "height_cm": [height_cm],
                "bodyweight_kg": [bodyweight_kg],
                "workout_name": [workout_name],
                "set_weight_kg": [set_weight_kg],
                "reps": [reps],
                "duration_min": [duration_min],
            }
        )
        try:
            pred = model.predict(input_df)[0]
            return float(pred)
        except Exception:
            return formula_calories(bodyweight_kg, workout_name, duration_min)
    else:
        return formula_calories(bodyweight_kg, workout_name, duration_min)

def suggest_weight_increase(current_weight: float, estimated_1rm: float, rpe: float) -> str:
    """
    Suggestion based on % of 1RM and RPE.
    """
    if current_weight <= 0 or estimated_1rm <= 0:
        return "Not enough data to suggest. Check your inputs."

    ratio = current_weight / estimated_1rm  # how heavy vs your 1RM

    if ratio < 0.7 and rpe <= 7:
        return (
            "You are lifting quite comfortably (<70% of your estimated 1RM, RPE ≤ 7). "
            "You can safely increase the weight next session (e.g., +2.5–5 kg)."
        )
    elif 0.7 <= ratio <= 0.85 and 7 <= rpe <= 9:
        return (
            "You are in a solid working range (70–85% of 1RM, RPE 7–9). "
            "You can maintain this weight or make small increases if it feels good."
        )
    elif ratio > 0.85 or rpe >= 9:
        return (
            "You are close to your max effort (>85% of 1RM or RPE ≥ 9). "
            "Increase weight carefully, focus on form, or consider adding volume instead."
        )
    else:
        return (
            "Your set is in a moderate range. You can either increase weight slightly, "
            "or stay here and add more reps/sets depending on your goal."
        )

def append_history(row: dict):
    """
    Append one set's data to workout_history.csv
    """
    df_row = pd.DataFrame([row])
    if os.path.exists(HISTORY_FILE):
        df_row.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df_row.to_csv(HISTORY_FILE, mode="w", header=True, index=False)

def load_history() -> pd.DataFrame:
    """
    Load history CSV if exists, else return empty DataFrame
    """
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE, parse_dates=["timestamp"])
    else:
        return pd.DataFrame()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Workout ML Tracker", page_icon="🏋️")
st.title("🏋️ Workout Tracker – Calories, 1RM (RPE-based) & Progress Graph")

st.caption(model_status)

st.markdown(
    "Enter your workout details below. The app will:\n"
    "- Estimate **calories burned** for this set\n"
    "- Estimate **1 Rep Max (1RM)** using **weight + reps + RPE**\n"
    "- Give a **suggestion** for next session weight\n"
    "- Save your set to history and show a **graph of your progress**"
)

# -------- Workout selection (selectbox) --------
workout_options = [
    "Bench Press",
    "Squat",
    "Deadlift",
    "Overhead Press",
    "Barbell Row",
    "Lat Pulldown",
    "Pull Up",
    "Push Up",
    "Bicep Curl",
    "Tricep Pushdown",
    "Leg Press",
    "Leg Extension",
    "Leg Curl",
    "Lunge",
    "Other (type manually)"
]

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["M", "F"], index=0)
    height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=1.0)
    bodyweight_kg = st.number_input("Bodyweight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5)

with col2:
    selected_workout = st.selectbox("Workout", workout_options, index=0)
    if selected_workout == "Other (type manually)":
        workout_name = st.text_input("Enter workout name", "")
    else:
        workout_name = selected_workout

    set_weight_kg = st.number_input("Weight used in this set (kg)", min_value=1.0, max_value=400.0, value=40.0, step=0.5)
    reps = st.number_input("Reps in this set", min_value=1, max_value=50, value=8, step=1)

# RPE input
rpe = st.slider("RPE for this set (1 = very easy, 10 = all-out max)", min_value=6.0, max_value=10.0, value=8.0, step=0.5)

duration_min = st.number_input(
    "Approx duration for this set (minutes)",
    min_value=0.5,
    max_value=30.0,
    value=3.0,
    step=0.5,
)

if st.button("Calculate & Save"):
    if workout_name.strip() == "":
        st.error("Please enter a workout name.")
    else:
        # 1RM (RPE-based)
        one_rm = estimate_1rm_rpe(set_weight_kg, reps, rpe)

        # Calories burned
        calories = predict_calories(
            gender=gender,
            height_cm=height_cm,
            bodyweight_kg=bodyweight_kg,
            workout_name=workout_name,
            set_weight_kg=set_weight_kg,
            reps=reps,
            duration_min=duration_min,
        )

        # Suggestion
        suggestion = suggest_weight_increase(set_weight_kg, one_rm, rpe)

        # Save to history
        row = {
            "timestamp": datetime.now(),
            "gender": gender,
            "height_cm": height_cm,
            "bodyweight_kg": bodyweight_kg,
            "workout_name": workout_name,
            "set_weight_kg": set_weight_kg,
            "reps": reps,
            "rpe": rpe,
            "duration_min": duration_min,
            "estimated_1rm": one_rm,
            "calories": calories,
        }
        append_history(row)

        st.success("Set saved to history ✅")

        st.markdown("## 📊 Results")
        st.metric("Estimated calories burned (this set)", f"{calories:.1f} kcal")
        st.metric("Estimated 1 Rep Max (1RM, RPE-based)", f"{one_rm:.1f} kg")

        st.markdown("### 🧠 Weight progression suggestion")
        st.write(suggestion)

        st.markdown("### 📝 Input Summary")
        st.json(row)

# -------- History & Graph Section --------
st.markdown("---")
st.markdown("## 📈 Progress Graph")

history_df = load_history()

if history_df.empty:
    st.info("No history yet. Calculate at least one set to see your progress graph.")
else:
    # Let user choose which workout to visualize
    available_workouts = sorted(history_df["workout_name"].unique().tolist())
    selected_history_workout = st.selectbox(
        "Select workout to show graph",
        available_workouts,
        index=0,
        key="history_workout_select"
    )

    filtered = history_df[history_df["workout_name"] == selected_history_workout].sort_values("timestamp")

    st.write(f"Showing history for: **{selected_history_workout}**")

    # Create a small DataFrame for chart
    chart_df = filtered[["timestamp", "estimated_1rm", "calories"]].set_index("timestamp")

    st.line_chart(chart_df)

    st.dataframe(filtered[["timestamp", "workout_name", "set_weight_kg", "reps", "rpe", "estimated_1rm", "calories"]])

