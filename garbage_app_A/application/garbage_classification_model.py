import numpy as np
import os
import random

def load_model():
    # Construct the correct model path
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'garbage_classification_model.keras')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def classify_image(model, image):
    prediction = np.array([[random.uniform(0, 1) for i in range(6)]])
    predicted_class = np.argmax(prediction, axis=1)[0]
    print(prediction)
    class_labels = {0: "Food_Organics", 1:'Glass', 2:'Metal', 3:'Paper_Cardboard', 4:'Plastic', 5:'Textile Trash', 6:'Vegetation'}
    return class_labels.get(predicted_class, "Unknown")
