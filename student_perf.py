import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# MongoDB connection
uri = "mongodb+srv://gary:gary@cluster0.uoimqso.mongodb.net/?appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['student']
collection = db['student_pred']

def load_model():
    """Load the trained model, scaler, and label encoder from a pickle file."""
    with open("student_final_mdel.pkl", 'rb') as file:  # Corrected typo from 'student_final_mdel.pkl'
        model, scaler, le = pickle.load(file)
        return model, scaler, le
    
def preprocess_data(data, scaler, le):
    """Transform input data using scaler and label encoder."""
    # Convert extracurricular activities to numerical value
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict(data):
    """Make predictions using the loaded model."""
    model, scaler, le = load_model()
    processed_data = preprocess_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    """Main function to run the Streamlit application."""
    st.title('Student Performance Prediction')
    st.write("Enter your data to get a prediction")

    hr_study = st.number_input("Hours studied: ", min_value=1, max_value=10, value=5)
    prev_score = st.number_input("Previous score: ", min_value=30, max_value=100, value=50)
    extra_c = st.selectbox("Extracurricular Activities: ", ['Yes', 'No'])
    sleep_hr = st.number_input("Sleeping hour: ", min_value=4, max_value=10, value=5)
    num_ques = st.number_input("Questions solved: ", min_value=0, max_value=10, value=5)

    if st.button("Predict score"):
        user_data = {
            'Hours Studied': hr_study,
            'Previous Scores': prev_score,
            'Sleep Hours': sleep_hr,
            'Sample Question Papers Practiced': num_ques,
            'Extracurricular Activities': extra_c
        }

        try:
            result = predict(user_data)
            # Convert prediction to standard data types
            user_data['prediction'] = result.tolist()  # Convert array to list
            collection.insert_one(user_data)  # Insert into MongoDB
            st.success(f"Your predicted result is {result[0]:.2f}")  # Format output for display
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()