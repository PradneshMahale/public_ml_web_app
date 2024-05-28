# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:24:41 2024

@author: Pradnesh
"""

import numpy as np

import pickle

# loading the saved model

loaded_model=pickle.load(open("D:/Deploy ML Model/trained_model.sav",'rb'))


input_data=(4,110,92,0,0,37.6,0.191,30)

# change the input_data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)

print(prediction)

if(prediction[0]==0):
  print("The Person is not Diabetic")
else:
  print("The Person is Diabetic")