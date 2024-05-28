# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:27:48 2024

@author: Pradnesh
"""

import numpy as np
import pickle 
import streamlit as st

# loading the saved model

loaded_model=pickle.load(open("D:/Deploy ML Model/trained_model.sav",'rb'))

# creating a function for Prediction

def diabetes_prediction(input_data):

    # change the input_data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)

    print(prediction)

    if(prediction[0]==0):
      return "The Person is not Diabetic"
    else:
      return "The Person is Diabetic"
  
    
def main():
    
    # giving a title 
    st.title("Diabetes prediction Web App ")
    
    # getting the input data from the user
    
    
    Pregnancies=st.text_input("Number of pregnaccies")
    Glucose=st.text_input("Gluose Level")
    BloodPressure=st.text_input("Blood Pressure Value")
    SkinThickness=st.text_input("Skin Thickness value")
    Insulin=st.text_input("Insulin Level")
    BMI=st.text_input("BMI value")
    DiabetesPedigreeFunction=st.text_input("Diabetes Function value")
    Age=st.text_input("Age ot the person")
    
    # code for Prediction
    diagnosis=''
    
    # creating  a button for Prediction
    
    if st.button("Diabetes Test Result"):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()