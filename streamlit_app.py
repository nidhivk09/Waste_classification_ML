import os
import numpy as np

import streamlit as st

# Define the class labels
LABELS = [
    "Battery", "Biological", "Brown Glass", "Cardboard", "Clothes",
    "Green Glass", "Metal", "Paper", "Plastic", "Shoes", "Trash", "White Glass"
]


def main():
    st.title("Waste Classification Portal")
    st.subheader("Upload an image of waste to classify it into one of 12 categories.")
    
    # Option to train model
   

    # Upload and classify images
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        
if __name__ == "__main__":
    main()
