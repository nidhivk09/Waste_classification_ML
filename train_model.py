import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2

# Define the class labels
LABELS = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]

# Paths
DATASET_PATH = "dataset/"  # Ensure your dataset is here
MODEL_PATH = "waste_classifier_model_cnn.h5"

# Function to define and train the CNN model
def train_cnn_model():
    # Data Generators
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical",
        subset="training",
    )

    val_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
    )

    # CNN Architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(LABELS), activation='softmax')  # Output layer for 12 categories
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    history = model.fit(train_gen, validation_data=val_gen, epochs=13)

    # Save the model
    model.save(MODEL_PATH)
    print("Model training complete and saved as:", MODEL_PATH)

    # Plot training accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.show()

if __name__ == "__main__":
    train_cnn_model()
