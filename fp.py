from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# ---- load your saved best XGBoost model ----
with open("best1xgboost_model.pkl", "rb") as f:
    model = joblib.load(f)

# ---- label-encoding consistent with your notebook ----
MAP_GENDER = {"Female": 0, "Male": 1}                  # LabelEncoder sorts alphabetically
MAP_COMPANY = {"Product": 0, "Service": 1}
MAP_WFH = {"No": 0, "Yes": 1}

# column order used during training
TRAIN_COLS = [
    "Gender", "Company Type", "WFH Setup Available",
    "Designation", "Resource Allocation", "Mental Fatigue Score"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # --- read form ---
        gender = request.form.get("gender")
        company_type = request.form.get("company_type")
        wfh_setup = request.form.get("wfh_setup")
        mental_fatigue = request.form.get("mental_fatigue")
        resource_allocation = request.form.get("resource_allocation")
        designation = request.form.get("designation")

        # --- basic validation & casting ---
        errors = []
        if gender not in MAP_GENDER: errors.append("Select a valid Gender.")
        if company_type not in MAP_COMPANY: errors.append("Select a valid Company Type.")
        if wfh_setup not in MAP_WFH: errors.append("Select a valid WFH option.")

        try:
            mental_fatigue = float(mental_fatigue)
            if not (0 <= mental_fatigue <= 10): errors.append("Mental Fatigue must be 0–10.")
        except:
            errors.append("Mental Fatigue must be a number.")

        try:
            resource_allocation = float(resource_allocation)
            if not (1 <= resource_allocation <= 10): errors.append("Resource Allocation must be 1–10.")
        except:
            errors.append("Resource Allocation must be a number.")

        try:
            designation = float(designation)
            if not (1 <= designation <= 5): errors.append("Designation must be 1–5.")
        except:
            errors.append("Designation must be a number.")

        if errors:
            return render_template(
                "index.html",
                prediction_text=" | ".join(errors),
                gender=gender,
                company_type=company_type,
                wfh_setup=wfh_setup,
                designation=request.form.get("designation"),
                resource_allocation=request.form.get("resource_allocation"),
                mental_fatigue=request.form.get("mental_fatigue")
            )

        # --- encode categoricals as in notebook  
        row = {
            "Gender": gender,
            "Company Type": company_type,
            "WFH Setup Available": wfh_setup,
            "Designation": designation,
            "Resource Allocation": resource_allocation,
            "Mental Fatigue Score": mental_fatigue
        }

        # IMPORTANT: keep the same column order as training
        X_infer = pd.DataFrame([[row[c] for c in TRAIN_COLS]], columns=TRAIN_COLS)

        # --- predict ---
        burn_rate = float(model.predict(X_infer)[0])

        # --- risk label for styling ---
        if burn_rate < 0.4:
            risk_level = "low"
            suggestion = "Low burnout risk — maintain current balance."
        elif burn_rate < 0.6:
            risk_level = "medium"
            suggestion = "Moderate risk — consider breaks or workload adjustments."
        elif burn_rate < 0.75:  
             risk_level = "elevated"
             suggestion = "Elevated risk — actively monitor workload and provide support."    
        else:
            risk_level = "high"
            suggestion = "High risk — prioritize immediate intervention."

        return render_template(
            "index.html",
            prediction_text=f"Predicted Burn Rate: {burn_rate:.2f}",
            suggestion_text=suggestion,
            risk_level=risk_level,
            gender=gender,
            company_type=company_type,
            wfh_setup=wfh_setup,
            designation=request.form.get("designation"),
            resource_allocation=request.form.get("resource_allocation"),
            mental_fatigue=request.form.get("mental_fatigue")
        )
   
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            gender=request.form.get("gender"),
            company_type=request.form.get("company_type"),
            wfh_setup=request.form.get("wfh_setup"),
            designation=request.form.get("designation"),
            resource_allocation=request.form.get("resource_allocation"),
            mental_fatigue=request.form.get("mental_fatigue")
        )                                         
if __name__ == "__main__":
    app.run(debug=True)
        