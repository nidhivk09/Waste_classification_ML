
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define the class labels
LABELS = [
    "Battery", "Biological", "Brown Glass", "Cardboard", "Clothes",
    "Green Glass", "Metal", "Paper", "Plastic", "Shoes", "Trash", "White Glass"
]

# Streamlit App
def main():
    st.title("Waste Classification Portal")
    st.subheader("Upload an image of waste to classify it into one of 12 categories.")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")
        
        # Load the trained model
        model = load_model("waste_classifier_model.h5")

        # Preprocess the image
        img = image.resize((128, 128))  # Ensure image is 128x128
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        predicted_label = LABELS[np.argmax(predictions)]

        st.write(f"### Prediction: {predicted_label}")

if __name__ == "__main__":
    main()



# Paths
DATASET_PATH = "dataset/"
MODEL_SAVE_PATH = "waste_classifier_model.h5"

# Data Generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')  # Output classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save Model
model.save(MODEL_SAVE_PATH)
