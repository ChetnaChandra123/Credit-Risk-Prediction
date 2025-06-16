import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

# Load models and scaler
trained_models = {
    'Logistic Regression': joblib.load("logistic_regression_model.pkl"),
    'Random Forest': joblib.load("random_forest_model.pkl"),
    'Decision Tree': joblib.load("decision_tree_model.pkl"),
    'SVM': joblib.load("svm_model.pkl"),
    'Naive Bayes': joblib.load("naive_bayes_model.pkl"),
}
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# PATCH for missing attributes
for model in trained_models.values():
    if isinstance(model, RandomForestClassifier):
        for tree in model.estimators_:
            if not hasattr(tree, 'monotonic_cst'):
                tree.monotonic_cst = None
    if isinstance(model, DecisionTreeClassifier):
        if not hasattr(model, 'monotonic_cst'):
            model.monotonic_cst = None


# Streamlit UI
st.title("Credit Risk Prediction")

# User's Name Input
user_name = st.text_input("Enter your name:")

# Input fields for features
age = st.number_input("Age", min_value=18, max_value=100)
sex = st.selectbox("Sex", ["Male", "Female"])
job = st.number_input("Job", min_value=0, max_value=4)
housing = st.selectbox("Housing", ["Own", "Rent", "Free"])
saving_accounts = st.selectbox("Saving Accounts", ["Little", "Moderate", "Rich", "Quite Rich"])
checking_account = st.selectbox("Checking Account", ["Little", "Moderate", "Rich"])
credit_amount = st.number_input("Credit Amount", min_value=1000, max_value=50000)
duration = st.number_input("Duration", min_value=1, max_value=72)
purpose = st.selectbox("Purpose", ["new car", "used car", "business", "education", "household", "others"])

# Prepare input
input_data = {
    'Age': age,
    'Sex': sex,
    'Job': job,
    'Housing': housing,
    'Saving accounts': saving_accounts,
    'Checking account': checking_account,
    'Credit amount': credit_amount,
    'Duration': duration,
    'Purpose': purpose
}

def recommend_credit_improvement(predictions):
    # Count how many models predict "good" vs "bad"
    good_count = predictions.count('good')
    bad_count = predictions.count('bad')

    # Recommendations based on predictions
    if good_count > bad_count:
        return [
            "Your credit is in good standing. However, maintaining a good credit history is crucial.",
            "We recommend continuing to save regularly and manage your expenses efficiently.",
            "It's important to monitor your credit usage and maintain timely payments."
        ]
    elif bad_count > good_count:
        return [
            "Your credit risk is higher. To improve your credit score, we suggest the following:",
            "Consider reducing your credit card debt and increasing your savings.",
            "Aim to pay off outstanding loans to reduce your overall debt.",
            "Focus on lowering your debt-to-income ratio and avoid taking out new loans.",
            "Make timely payments on all your credit accounts to build a positive payment history."
        ]
    else:
        return [
            "Your credit risk is somewhat balanced. Consider the following actions to strengthen your financial standing:",
            "Keep your credit utilization low and avoid accumulating unnecessary debt.",
            "Regularly monitor your credit reports for any discrepancies or areas for improvement.",
            "Maintaining a good mix of credit types can also help improve your creditworthiness."
        ]

def predict_credit_risk(new_data_dict):
    new_data = pd.DataFrame([new_data_dict])
    new_data_encoded = pd.get_dummies(new_data)

    # Align with training columns
    missing_cols = set(feature_columns) - set(new_data_encoded.columns)
    for col in missing_cols:
        new_data_encoded[col] = 0
    new_data_encoded = new_data_encoded[feature_columns]

    # Remove any unexpected extra columns not in training
    extra_cols = set(new_data_encoded.columns) - set(feature_columns)
    new_data_encoded.drop(columns=extra_cols, inplace=True)


    # Scale
    new_data_scaled = scaler.transform(new_data_encoded)

     # Predictions from all models
    # predictions = []
    # for name, model in trained_models.items():
    #     if hasattr(model, "predict"):
    #         prediction = model.predict(new_data_scaled)
    #         predicted_class = label_encoder.inverse_transform(prediction)
    #         predictions.append(predicted_class[0])

    # Predictions from all models
    
    predictions = []
    model_predictions = {}
    
    for name, model in trained_models.items():
        if hasattr(model, "predict"):
            try:
                prediction = model.predict(new_data_scaled)
                predicted_class = label_encoder.inverse_transform(prediction)
                predictions.append(predicted_class[0])
                model_predictions[name] = predicted_class[0]  # Save for UI display
            except Exception as e:
                model_predictions[name] = f"Error: {e}"
    
    # Majority Voting: Get the most common prediction
    majority_vote = Counter(predictions).most_common(1)[0][0]
    
    # Recommendations based on majority vote
    recommendations = recommend_credit_improvement(predictions)

    # return majority_vote, recommendations
    return majority_vote, recommendations, model_predictions


# Button
if st.button("Predict"):
    if user_name:  # Ensure name is provided
        # final_prediction, recommendations = predict_credit_risk(input_data)
        final_prediction, recommendations, model_predictions = predict_credit_risk(input_data)
        # Display all model predictions
        
        st.subheader("Model-wise Predictions")
        for model_name, model_pred in model_predictions.items():
            st.write(f"{model_name}: {model_pred}")


        # Display greeting and results
        #st.write(f"Hello {user_name}, your credit risk prediction is: {final_prediction}\n")

        if final_prediction == 'good':
            st.success(f"Hello {user_name}, your credit risk prediction is: {final_prediction}")
        else:
            st.error(f"Hello {user_name}, your credit risk prediction is: {final_prediction}")

        
        # Show recommendations
        for rec in recommendations:
            st.write(f"• {rec}")
        
        st.write("\nThank you for using the Credit Risk Prediction tool. We wish you the best in managing your financial health.")
    else:
        st.error("Please enter your name!")
