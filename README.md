
# Credit Card Fraud Detection Using Random Forest

This project is an end-to-end machine learning application that detects fraudulent credit card transactions using the Random Forest algorithm. It includes data preprocessing, model training, a Flask-based web interface, and real-time fraud prediction.

---

## 📌 Features

- 📊 Uses Random Forest Classifier for high-accuracy prediction
- 🧪 Handles class imbalance with SMOTE
- 🧠 Trained on real-world anonymized credit card transaction data
- 🌐 Flask web interface with live prediction
- 🎨 User-friendly and responsive UI
- ✅ Deployable on platforms like Render or Heroku

---

## 🔍 Dataset

- Source: [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains: 284,807 transactions
- Features: V1–V28 (PCA-transformed), `Amount`
- Label: `Class` (0 = Not Fraud, 1 = Fraud)

---

## ⚙️ Project Structure

```

credit\_card\_fraud\_detection/
│
├── dataset/
│   └── creditcard.csv                 # Input dataset
├── models/
│   └── fraud\_model.pkl               # Trained model (saved with joblib)
├── static/
│   └── style.css                     # CSS for frontend
├── templates/
│   └── index.html                    # Web UI template
├── app.py                            # Flask API and Web App
├── main.py                           # Training script
└── README.md                         # Project documentation

````

---

## 🧪 How It Works

1. **main.py**
   - Loads and cleans the dataset
   - Scales features and applies SMOTE
   - Trains a Random Forest model
   - Saves the trained model as `fraud_model.pkl`

2. **app.py**
   - Loads the saved model
   - Exposes a `/predict` route via Flask
   - Accepts JSON input or form data
   - Returns prediction result as JSON or HTML

3. **Web UI**
   - User inputs transaction details (V1–V28, Amount)
   - Receives real-time prediction: **Fraud** or **Not Fraud**

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
````

### 2. Create a virtual environment and activate it

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python main.py
```

### 5. Run the Flask app

```bash
python app.py
```

### 6. Open in browser

Go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ✅ Future Improvements

* Add login and transaction history
* Use advanced models like XGBoost or Isolation Forest
* Deploy with authentication (OAuth, JWT)
* Integrate with real-time transaction APIs

---

## 👨‍💻 Authors

* Project by: *Akshaya Harithsa*
* Institution: *SJB Institute of Technology*


