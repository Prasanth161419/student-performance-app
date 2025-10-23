import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

model=joblib.load("best_model.pkl")
st.title("Student's performance predictor")

study_hours=st.slider("Study Hours per Day",0.0,12.0,2.0)
attendance=st.slider("Attendance percentage",0.0,100.0,80.0)
mental_health=st.slider("Mental health Rating (1-10)",1,10,5)
sleep_hours=st.slider("Sleep Hours per Night",0.0,12.0,7.0)
part_time_job=st.selectbox("part-Time job",["No","yes"])

ptj_encoded=1 if part_time_job=="yes" else 0

if st.button("predict"):
    input_data=np.array([[study_hours,attendance,mental_health,sleep_hours,ptj_encoded]])
    prediction=model.predict(input_data)[0]

    prediction=max(0,min(100,prediction))

    st.success(f"predicted performance:{prediction:.2f}")

