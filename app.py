import random
import streamlit as st
import torch
import numpy as np
from model import ColourClassifier
import pickle


# load model
model = ColourClassifier()
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(r, g, b):
    input = torch.tensor([[r/255, g/255, b/255]], dtype=torch.float32)
    model.eval()
    with torch.no_grad(): # disable gradient calculation
        prediction = model(input)
    return(prediction > 0.5).float().item()

def generate_muted_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    if predict(r, g, b):
        return r, g, b
    return generate_muted_color()

def display_colour(r, g, b):
    color_hex = f"#{r:02x}{g:02x}{b:02x}"  # convert RGB to Hex
    st.markdown(f'<div style="background-color:{color_hex}; width: 100px; height: 100px; display: inline-block; margin: 5px;"></div>', unsafe_allow_html=True)
    st.write(f'RGB Value: ({r}, {g}, {b})')
    st.write(f'Hex Value: #{r:02x}{g:02x}{b:02x}')

def display_colours():
    cols = st.columns(5)
    for i in range(5):
        r, g, b = generate_muted_color()
        cols[i].markdown(f'<div style="background-color: #{r:02x}{g:02x}{b:02x}; width: 100px; height: 100px; display: inline-block; margin: 5px; border-radius: 50%;"></div>', unsafe_allow_html=True)
        cols[i].write(f'RGB: ({r}, {g}, {b})')
        cols[i].write(f'Hex: #{r:02x}{g:02x}{b:02x}')
        #display_colour(r, g, b)

st.title('Generate a Random Muted Colour Palette')

# # predict if a colour is muted or not

# # rgb sliders
# r1 = st.slider('Red', 0, 255, 30)
# g1 = st.slider('Green', 0, 255, 30)
# b1 = st.slider('Blue', 0, 255, 30)

# # display colour
# display_colour(r1, g1, b1)

# # make prediction
# if st.button('Predict'):
#     prediction = predict(r1, g1, b1)
#     st.write(f'Prediction: {"Muted" if prediction == 1 else "Not Muted"}') 


# generate random muted colour palatte

if st.button('Generate Palette'):
    display_colours()