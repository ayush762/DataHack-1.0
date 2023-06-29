import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import h5py
import xgboost as xgb


def predict_damage(image):
    if image is not None:
        image = Image.open(image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image = np.array(image)
        model = keras.models.load_model("D:\Data1\pages\\model.h5")
        # model2 = keras.models.load_model('D:\Data1\pages\xgboost_model.h5')
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = np.reshape(image, (1, 256, 256, 3))
        preds = model.predict(image)
        rounded_predictions = np.argmax(preds, axis=1)

        if rounded_predictions is 0:
            st.write("Damaged")
        else:
            st.write("Car is not Damaged")


Year = st.text_input("How Much Old is Your Car")

Kilometers = st.text_input("Kilometers Driven")

Mileage = st.text_input("Mileage")

Engine = st.text_input("Engine (in CC)")

Power = st.text_input("Power")

No_Of_Seats = st.text_input("No. Of Seats")

options = ['CNG', 'Diesel', 'LPG', 'Petrol']
selected_fuel = st.selectbox('Fuel:', options)

options = ['Manual', 'Automatic']
selected_operation = st.selectbox('Transmission:', options)

options = ['1st_Owner', '2nd_Owner', '3rd_Owner', '4th_Owner']
selected_hand = st.selectbox('Owner Type:', options)

options = ['', '', '', '']
selected_hand = st.selectbox('Owner Type:', options)

if selected_fuel is 'CNG':
    selected_fuel_Diesel = 0
    selected_fuel_LPG = 0
    selected_fuel_Petrol = 0
if selected_fuel is 'Diesel':
    selected_fuel_Diesel = 1
    selected_fuel_LPG = 0
    selected_fuel_Petrol = 0
if selected_fuel is 'LPG':
    selected_fuel_Diesel = 0
    selected_fuel_LPG = 1
    selected_fuel_Petrol = 0
if selected_fuel is 'Petrol':
    selected_fuel_Diesel = 0
    selected_fuel_LPG = 0
    selected_fuel_Petrol = 1

if selected_operation is 'Manual':
    selected_operation_Manual = 1
if selected_operation is 'Automatic':
    selected_operation_Manual = 0

if selected_hand is '1st_Owner':
    selected_hand_2nd_Owner = 0
    selected_hand_3rd_Owner = 0
    selected_hand_4th_Owner = 0
if selected_hand is '2nd_Owner':
    selected_hand_2nd_Owner = 1
    selected_hand_3rd_Owner = 0
    selected_hand_4th_Owner = 0
if selected_hand is '3rd_Owner':
    selected_hand_2nd_Owner = 0
    selected_hand_3rd_Owner = 1
    selected_hand_4th_Owner = 0
if selected_hand is '4th_Owner':
    selected_hand_2nd_Owner = 0
    selected_hand_3rd_Owner = 0
    selected_hand_4th_Owner = 1

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"])

predict_damage(uploaded_file)
# Load the XGBoost model from the .h5 file
# with h5py.File("D:\Data1\pages\xgboost_model.h5", 'r') as f:
#     booster = xgb.Booster()
#     booster.load_model(f)

# input_data = np.array([[Year, Kilometers, Mileage, Engine, Power, No_Of_Seats, selected_fuel_Diesel, selected_fuel_LPG,
#                       selected_fuel_Petrol, selected_operation_Manual, selected_hand_4th_Owner, selected_hand_2nd_Owner, selected_hand_3rd_Owner]])

# # Use the loaded model to predict the output based on the input parameters
# output = xgb.Booster.predict(xgb.DMatrix(input_data))

# st.write('Output:', output[0])


def predict_damage(image):
    if image is not None:
        image = Image.open(image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image = np.array(image)
        model = keras.models.load_model("D:\Data1\pages\model.h5")
        # model2 = keras.models.load_model('D:\Data1\pages\xgboost_model.h5')
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = np.reshape(image, (1, 256, 256, 3))
        preds = model.predict(image)
        rounded_predictions = np.argmax(preds, axis=1)

        if rounded_predictions is 0:
            st.write("Sorry your Car is Damaged.")
        else:
            st.write("Your car is in perfect condition.")
            
        
