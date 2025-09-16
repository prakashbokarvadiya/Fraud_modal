from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model & encoder
model = joblib.load("fraud_model.pkl")
encoder = joblib.load("encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    df = pd.DataFrame([{
        "step": int(data["step"]),
        "amount": float(data["amount"]),
        "oldbalanceOrg": float(data["oldbalanceOrg"]),
        "newbalanceOrig": float(data["newbalanceOrig"]),
        "oldbalanceDest": float(data["oldbalanceDest"]),
        "newbalanceDest": float(data["newbalanceDest"]),
        "type": data["type"]
    }])

    # OneHotEncoding
    df_encoded = encoder.transform(df[["type"]]).toarray()
    df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(["type"]))
    df_final = pd.concat([df.drop("type", axis=1), df_encoded], axis=1)

    prediction = model.predict(df_final)[0]
    result = "🚨 Fraud Detected!" if prediction == 1 else "✅ Legitimate Transaction"

    return render_template("index.html", prediction=result)




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render ka PORT use kare
    app.run(host="0.0.0.0", port=port, debug=True)
