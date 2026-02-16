from flask import Flask, render_template, request
import numpy as np
import pickle
import json

app = Flask(__name__)


# ---------------- MODEL LOADER ----------------
def load_model(model_choice):

    if model_choice == "knn":
        with open("Iris_k-NN_model.pkl", "rb") as f:
            model = pickle.load(f)

    elif model_choice == "nb":
        with open("Iris_NB_model.pkl", "rb") as f:
            model = pickle.load(f)

    else:
        return None

    return model


# ---------------- METRICS LOADER ----------------
def load_metrics(model_choice):

    if model_choice == "knn":
        report_file = "classification_report_kNN.json"
        cm_file = "confusion_matrix_kNN.json"

    elif model_choice == "nb":
        report_file = "classification_report_NB.json"
        cm_file = "confusion_matrix_NB.json"

    else:
        return None, None, None

    # load classification report
    with open(report_file, "r") as f:
        report_raw = json.load(f)

    # load confusion matrix
    with open(cm_file, "r") as f:
        cm = json.load(f)

    # -------- normalize classification report --------
    report_rows = []
    for label, metrics in report_raw.items():
        if isinstance(metrics, dict):
            report_rows.append({
                "label": label,
                "precision": round(metrics.get("precision", 0), 3),
                "recall": round(metrics.get("recall", 0), 3),
                "f1": round(metrics.get("f1-score", 0), 3),
                "support": int(metrics.get("support", 0))
            })

    accuracy = round(report_raw.get("accuracy", 0), 3)

    # ensure confusion matrix is 2D list
    if isinstance(cm, dict):
        cm = [list(row.values()) for row in cm.values()]

    return report_rows, accuracy, cm


# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template('index.html')


# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():

    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width  = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width  = float(request.form['petal_width'])
    except ValueError:
        return "Invalid numeric input", 400

    model_choice = request.form.get('model', '').strip().lower()

    if model_choice not in ["knn", "nb"]:
        return "Invalid model selection", 400

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # load model only when needed
    model = load_model(model_choice)
    if model is None:
        return "Model not found", 500

    # predict
    pred_class = int(model.predict(features)[0])

    species_map = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    prediction = species_map.get(pred_class, "Unknown")

    # load metrics
    report_rows, accuracy, cm = load_metrics(model_choice)

    return render_template(
        'index.html',
        prediction=prediction,
        model_used=model_choice.upper(),
        report_rows=report_rows,
        accuracy=accuracy,
        confusion_matrix=cm
    )


# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)
