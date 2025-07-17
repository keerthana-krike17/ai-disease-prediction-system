# 🧠 AI-Driven Disease Prediction System

A smart and interactive web application that predicts the risk of heart disease using machine learning, explains predictions using SHAP, and generates downloadable PDF medical reports. Built with Flask, XGBoost, SHAP, SQLite, and HTML/CSS.

---

## 🚀 Features

- ✅ **Login & User Authentication**
- 📊 **Heart Disease Prediction** using trained ML model (XGBoost)
- 🧪 **Input Form** with medical attributes (age, cholesterol, etc.)
- 🌈 **SHAP Explainability** to visualize feature impact
- 📝 **Auto-generated PDF Reports**
- 📁 **Patient History Dashboard**
- 📦 Clean, modular backend and frontend code structure

---

## 📷 Screenshots

| Prediction Form | SHAP Explanation | PDF Report |
|-----------------|------------------|-------------|
| ![Form](screenshots/form.png) | ![SHAP](screenshots/shap.png) | ![PDF](screenshots/pdf.png) |

---

## 🧬 ML Model Details

- **Algorithm**: XGBoost Classifier
- **Preprocessing**: One-hot encoding, scaling with `StandardScaler`
- **Training Dataset**: Public heart disease dataset (UCI)
- **Accuracy**: ~88% on test data

---

## 🛠️ Technologies Used

| Category      | Tools/Libraries                       |
|---------------|----------------------------------------|
| Frontend      | HTML, CSS, Bootstrap                  |
| Backend       | Flask, Python                         |
| ML Model      | XGBoost, Scikit-learn, Pandas         |
| Explainability| SHAP                                  |
| PDF Reports   | FPDF                                  |
| Database      | SQLite                                |

---

## 📂 Project Structure
```
ai-disease-prediction-system/
│
├── app.py
├── model.py
├── reset_db.py
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── features.pkl
├── shap_explainer.pkl
├── heart.csv
├── utils/
│ └── report_generator.py
├── templates/
│ ├── login.html
│ ├── register.html
│ ├── index.html
│ ├── dashboard.html
│ └── result.html
├── static/
│ ├── shap_1.png
│ └── images/
|  |──happy.gif
|  └──warning.gif
└── README.md
```

---

## 🔧 Installation Instructions

1. **Clone the Repo**
```bash
git clone https://github.com/yourusername/ai-disease-prediction-system.git
cd ai-disease-prediction-system
```
Create Virtual Environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install Dependencies
```
pip install -r requirements.txt
```
Initialize Database
```
python reset_db.py
```
Run App
```
python app.py
```
Then go to: http://127.0.0.1:5000
