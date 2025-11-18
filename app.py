from flask import Flask, render_template, request, jsonify
import pickle
import math
import os

app = Flask(__name__)

# ---- Load model ----
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl not found in project root. Place your trained model as model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---- Preprocessing (match your training pipeline) ----
def preprocess_data(gender, married, dependents, education, employed, credit, area,
                    ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term):
    # Basic encoding matching earlier examples
    male = 1 if str(gender).strip().lower() == "male" else 0
    married_yes = 1 if str(married).strip().lower() == "yes" else 0

    if str(dependents) in ("1", "1.0"):
        d1, d2, d3 = 1,0,0
    elif str(dependents) in ("2","2.0"):
        d1, d2, d3 = 0,1,0
    elif str(dependents).startswith("3"):
        d1, d2, d3 = 0,0,1
    else:
        d1, d2, d3 = 0,0,0

    not_grad = 1 if str(education).strip().lower() == "not graduate" else 0
    employed_yes = 1 if str(employed).strip().lower() == "yes" else 0
    semiurban = 1 if str(area).strip().lower() == "semiurban" else 0
    urban = 1 if str(area).strip().lower() == "urban" else 0

    # numeric conversions with safe fallback
    try:
        ApplicantIncome = float(ApplicantIncome)
    except:
        ApplicantIncome = 0.0
    try:
        CoapplicantIncome = float(CoapplicantIncome)
    except:
        CoapplicantIncome = 0.0
    try:
        LoanAmount = float(LoanAmount)
    except:
        LoanAmount = 1.0
    try:
        Loan_Amount_Term = float(Loan_Amount_Term)
    except:
        Loan_Amount_Term = 360.0

    ApplicantIncomeLog = math.log(ApplicantIncome) if ApplicantIncome > 0 else 0.0
    TotalIncomeLog = math.log(ApplicantIncome + CoapplicantIncome) if (ApplicantIncome + CoapplicantIncome) > 0 else 0.0
    LoanAmountLog = math.log(LoanAmount) if LoanAmount > 0 else 0.0
    LoanTermLog = math.log(Loan_Amount_Term) if Loan_Amount_Term > 0 else 0.0

    try:
        credit_val = float(credit)
    except:
        credit_val = 0.0
    credit_flag = 1 if 800 <= credit_val <= 1000 else 0

    # feature vector order must match training
    features = [
        credit_flag,
        ApplicantIncomeLog,
        LoanAmountLog,
        LoanTermLog,
        TotalIncomeLog,
        male,
        married_yes,
        d1, d2, d3,
        not_grad,
        employed_yes,
        semiurban,
        urban
    ]
    return features

# ---- Routes ----
@app.route("/")
def home():
    # content text stored in template
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        form = request.form
        try:
            features = preprocess_data(
                form.get("gender","Male"),
                form.get("married","No"),
                form.get("dependents","0"),
                form.get("education","Graduate"),
                form.get("employed","No"),
                form.get("credit","750"),
                form.get("area","Urban"),
                form.get("ApplicantIncome","5000"),
                form.get("CoapplicantIncome","0"),
                form.get("LoanAmount","100"),
                form.get("Loan_Amount_Term","360")
            )
            pred = model.predict([features])[0]
            # handle bytes/string label
            if isinstance(pred, bytes):
                pred = pred.decode()

            approved = str(pred).strip().upper() in ("Y","YES","1","APPROVED","APPROVE")
            status = "Approved" if approved else "Rejected"
            provided = dict(form)
            return render_template("predict.html", result=True, status=status, provided=provided)
        except Exception as e:
            # show error message in status
            return render_template("predict.html", result=True, status=f"Error: {e}", provided={})
    # GET
    return render_template("predict.html", result=False)

# Chatbot: accepts JSON answers and returns prediction
@app.route("/chat")
def chat_page():
    return render_template("chatbot.html")

@app.route("/chat_predict", methods=["POST"])
def chat_predict():
    data = request.json or {}
    try:
        features = preprocess_data(
            data.get("gender","Male"),
            data.get("married","No"),
            data.get("dependents","0"),
            data.get("education","Graduate"),
            data.get("employed","No"),
            data.get("credit","750"),
            data.get("area","Urban"),
            data.get("ApplicantIncome","5000"),
            data.get("CoapplicantIncome","0"),
            data.get("LoanAmount","100"),
            data.get("Loan_Amount_Term","360")
        )
        pred = model.predict([features])[0]
        if isinstance(pred, bytes):
            pred = pred.decode()
        approved = str(pred).strip().upper() in ("Y","YES","1","APPROVED","APPROVE")
        return jsonify({"result": ("Eligible for loan" if approved else "Not eligible"), "approved": approved})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
