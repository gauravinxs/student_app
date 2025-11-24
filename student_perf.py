import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_model():
    with open("student_final_mdel.pkl", 'rb') as file:
        model, scaler, le = pickle.load(file)
        return model, scaler, le
    

def preprocess_data(data, scaler, le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed


def pred(data):
    model, scaler, le = load_model()
    process_data = preprocess_data(data, scaler, le)
    pred = model.predict(process_data)
    return pred


def main():
    st.title('Student Performance Pred')
    st.write("Enter your data to get prediction")
    hr_study = st.number_input("Hours studied: ", min_value =1, max_value = 10, value =5)
    prev_score = st.number_input("Previous score: ", min_value =30, max_value = 100, value =50)
    extra_c = st.selectbox("Extra curri", ['Yes', 'No'])
    sleep_hr = st.number_input("Sleeping hour: ", min_value =4, max_value = 10, value =5)
    num_ques = st.number_input("Ques solved: ", min_value =0, max_value = 10, value =5)

    if st.button("Predict score"):
        user_data = {
            'Hours Studied':hr_study,
            'Previous Scores':prev_score,
            'Sleep Hours':sleep_hr,
            
            'Sample Question Papers Practiced': num_ques,
            'Extracurricular Activities': extra_c
        }
        res = pred(user_data)
        st.success(f"Your pred res is {res}")

if __name__ == "__main__":
    main()