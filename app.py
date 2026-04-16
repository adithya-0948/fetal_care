from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# ================= LOAD MODEL ================= #
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Label mapping
label_map = {
    1: "Normal",
    2: "Suspect",
    3: "Pathological"
}

# Feature names (from dataset)
feature_names = [
    'baseline value', 'accelerations', 'fetal_movement',
    'uterine_contractions', 'light_decelerations',
    'severe_decelerations', 'prolongued_decelerations',
    'abnormal_short_term_variability',
    'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'mean_value_of_long_term_variability', 'histogram_width',
    'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
    'histogram_number_of_zeroes', 'histogram_mode',
    'histogram_mean', 'histogram_median', 'histogram_variance',
    'histogram_tendency'
]

# ================= INIT RECORD FILE ================= #
if not os.path.exists("records.csv"):
    pd.DataFrame(columns=[
        "Patient_ID", "Name", "Gestational_Age",
        "Prediction", "Confidence",
        "Prob_Normal", "Prob_Suspect", "Prob_Pathological"
    ]).to_csv("records.csv", index=False)


# ================= HOME ================= #
@app.route('/')
def home():
    return render_template("index.html")


# ================= DASHBOARD ================= #
@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")


# ================= PREDICTION PAGE ================= #
@app.route('/predict')
def predict_page():
    return render_template("predict.html", features=feature_names)


# ================= PREDICTION ================= #
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Patient info
        name = request.form['name']
        pid = request.form['patient_id']
        age = request.form['age']

        # CTG features
        features = [float(x) for x in request.form.getlist('features')]

        if len(features) != 21:
            return "Error: 21 features required!"

        # Scale input
        scaled = scaler.transform([features])

        # Prediction
        pred = model.predict(scaled)[0]
        probs = model.predict_proba(scaled)[0]

        result = label_map[pred]
        confidence = round(max(probs) * 100, 2)

        # Save record
        record = pd.DataFrame([[
            pid, name, age, result, confidence,
            probs[0], probs[1], probs[2]
        ]], columns=[
            "Patient_ID", "Name", "Gestational_Age",
            "Prediction", "Confidence",
            "Prob_Normal", "Prob_Suspect", "Prob_Pathological"
        ])

        record.to_csv("records.csv", mode='a', header=False, index=False)

        return render_template(
            "result.html",
            result=result,
            confidence=confidence,
            probs={
                "Normal": round(probs[0]*100, 2),
                "Suspect": round(probs[1]*100, 2),
                "Pathological": round(probs[2]*100, 2)
            }
        )

    except Exception as e:
        return f"Error: {str(e)}"


# ================= RECORDS ================= #
@app.route('/records')
def records():
    df = pd.read_csv("records.csv")
    return render_template(
        "records.html",
        tables=df.values,
        columns=df.columns
    )


# ================= ANALYTICS ================= #
@app.route('/analytics')
def analytics():
    df = pd.read_csv("records.csv")

    # Counts for charts
    counts = df["Prediction"].value_counts().to_dict()

    # Confidence average
    avg_conf = round(df["Confidence"].mean(), 2) if len(df) > 0 else 0

    # Feature importance
    importances = model.feature_importances_
    feature_importance = dict(zip(feature_names, importances))

    # Sort top 10
    top_features = dict(sorted(feature_importance.items(),
                               key=lambda x: x[1],
                               reverse=True)[:10])

    return render_template(
        "analytics.html",
        counts=counts,
        avg_conf=avg_conf,
        features=list(top_features.keys()),
        values=list(top_features.values())
    )


# ================= DELETE RECORD ================= #
@app.route('/delete/<pid>')
def delete(pid):
    df = pd.read_csv("records.csv")
    df = df[df["Patient_ID"] != pid]
    df.to_csv("records.csv", index=False)
    return redirect(url_for('records'))


# ================= RUN ================= #
if __name__ == "__main__":
    app.run(debug=False)