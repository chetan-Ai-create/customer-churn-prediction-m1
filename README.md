# 📊 Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to churn (leave the company) based on their personal and service-related information.

## 🎯 Project Objective

Customer churn is an important problem for telecom companies. The goal of this project is to build a Machine Learning model that can identify customers who are likely to leave, so that the company can take preventive actions.

## 📊 Dataset

This project uses the IBM Telco Customer Churn dataset.

The dataset contains information about:
- Customer demographics
- Services used by customers
- Contract details
- Payment methods
- Monthly charges
- Total charges
- Customer churn status

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Removed the `customerID` column because it does not provide useful predictive information.
- Cleaned the `TotalCharges` column.
- Converted `TotalCharges` from object/string to numerical format.
- Handled missing values in `TotalCharges`.
- Converted the target variable `Churn`:
  - `No → 0`
  - `Yes → 1`
- Separated numerical and categorical features.
- Applied One-Hot Encoding to categorical features.
- Applied StandardScaler to numerical features.
- Used a train-test split with stratification.

## 🤖 Machine Learning Models

Two classification models were tested:

### 1. Logistic Regression

Logistic Regression was used as the primary classification model because it is simple, interpretable, and performs well for binary classification problems.

### 2. Random Forest

Random Forest was also tested to compare its performance with Logistic Regression.

## 📈 Model Results

### Logistic Regression

- Accuracy: **80.55%**
- ROC-AUC: **84.21%**

### Random Forest

- Accuracy: **78.57%**

Based on the evaluation results, Logistic Regression was selected as the final model.

## 🔍 Why Logistic Regression?

Logistic Regression performed better than Random Forest on this dataset based on the evaluated metrics.

Another advantage is interpretability. Its coefficients help understand which features are associated with higher or lower churn predictions.

For example:

- Higher tenure was associated with lower churn tendency.
- Month-to-month contracts were associated with higher churn tendency.
- Fiber optic internet service showed higher churn tendency in the trained model.

These relationships represent model patterns and should not be interpreted as direct causal relationships.

## 🔮 Prediction System

The trained model and preprocessing pipeline were saved using Joblib.

The application takes customer information such as:

- Gender
- Partner
- Dependents
- Internet Service
- Contract
- Payment Method
- Tenure
- Monthly Charges
- Total Charges

and predicts whether the customer is likely to churn.

The application also displays the estimated churn probability.

## 🌐 Streamlit Application

A Streamlit web application was created to provide an interactive interface for making predictions.

The user enters customer information and clicks **Predict Churn**.

The application then:

1. Creates a customer DataFrame.
2. Applies the saved preprocessing pipeline.
3. Sends the processed data to the trained Logistic Regression model.
4. Generates a churn prediction.
5. Displays the churn probability.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## 📁 Project Structure

```text
customer-churn-prediction-m1/
│
├── app.py
├── churn_model.pkl
├── preprocessor.pkl
├── requirements.txt
└── README.md

🚀 How to Run

1. Clone the repository

git clone https://github.com/chetan-AI-create/customer-churn-prediction-m1.git
cd customer-churn-prediction-m1
2. Install dependencies
pip install -r requirements.txt
3. Run the Streamlit application
streamlit run app.py

The application will open in your browser.

⚠️ Limitations
The model is trained on a specific telecom customer dataset.
Model predictions are based on patterns present in the training data.
Predictions should not be treated as guaranteed outcomes.
The dataset may not represent every telecom company's customer behavior.
🔮 Future Improvements
Try additional Machine Learning models.
Improve handling of class imbalance.
Perform hyperparameter tuning.
Add model monitoring.
Deploy the application publicly.
Add explainable AI features.
Build an AI assistant that can answer questions about this project using the repository's actual code and documentation.
👨‍💻 Author

Chetan

AI/ML Student | Building Machine Learning & Generative AI Projects
