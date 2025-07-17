from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
import pandas as pd
import numpy as np


import pickle
import shap
import os
from datetime import datetime
from utils.report_generator import generate_pdf_report
import matplotlib.pyplot as plt
from flask import send_file
from utils.report_generator import generate_pdf_report
import os
from utils.report_generator import generate_pdf_report




app = Flask(__name__)
app.secret_key = "your_secret_key"

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load ML components
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))
explainer = pickle.load(open("shap_explainer.pkl", "rb"))

# ---------- DB MODELS ----------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password = db.Column(db.String(80))

class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    timestamp = db.Column(db.String(100))
    input_data = db.Column(db.Text)
    result = db.Column(db.Integer)

# ---------- LOGIN MANAGER ----------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------- ROUTES ----------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            return "User already exists!"
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username'], password=request.form['password']).first()
        if user:
            login_user(user)
            return redirect('/dashboard')
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')

@app.route('/dashboard')
@login_required
def dashboard():
    logs = PredictionLog.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', logs=logs)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    input_dict = {k: float(v) if v.replace('.', '', 1).isdigit() else v for k, v in request.form.items()}
    df = pd.DataFrame([input_dict])

    # One-hot encode missing categories
    df = pd.get_dummies(df).reindex(columns=features, fill_value=0)
    X_scaled = scaler.transform(df)

    prediction = model.predict(X_scaled)[0]
    shap_values = explainer(df)[0].values

    # Save to DB
    log = PredictionLog(user_id=current_user.id,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        input_data=str(input_dict),
                        result=prediction)
    db.session.add(log)
    db.session.commit()

    # SHAP bar chart
    plt.figure(figsize=(8, 4))
    shap_values_bar = pd.Series(shap_values, index=features).nlargest(5)
    shap_values_bar.plot(kind='barh', color='skyblue')
    plt.title("Top 5 Feature Contributions")
    plt.xlabel("SHAP Value")
    plt.tight_layout()
    shap_path = f"static/shap_{current_user.id}.png"
    plt.savefig(shap_path)
    plt.close()

    return render_template("result.html", prediction=prediction,
                           shap_img=shap_path,
                           inputs=input_dict)


@app.route("/download_report")
@login_required
def download_report():
    try:
        log = PredictionLog.query.filter_by(user_id=current_user.id).order_by(PredictionLog.timestamp.desc()).first()

        if not log:
            flash("No prediction found to generate report.", "warning")
            return redirect(url_for("dashboard"))

        input_data = eval(log.input_data)
        result = log.result

        print("📄 Input Data:", input_data)
        print("📊 Result:", result)

        filename = f"report_{current_user.id}.pdf"
        file_path = os.path.join(os.getcwd(), filename)
        print("📂 File path:", file_path)

        # Generate the report
        generate_pdf_report(input_data, result, output_path=file_path, user_name=current_user.username)

        print("✅ PDF generated successfully!")

        return send_file(file_path, as_attachment=True)

    except Exception as e:
        print(f"❌ Error generating or sending PDF: {e}")
        flash("Something went wrong while generating the report.", "danger")
        return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=True)
