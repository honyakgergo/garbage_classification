import tensorflow as tf
import numpy as np
import os

def load_model():
    # Ensure the path points to the correct location inside the container
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'garbage_classification_model.keras')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def predict_class(model, image):
    prediction = model.predict(image)
    print(prediction)
    predicted_class = np.argmax(prediction, axis=1)[0]

    class_labels = {0: "Food_Organics", 1:'Glass', 2:'Metal', 3:'Paper_Cardboard', 4:'Plastic', 5:'Textile Trash', 6:'Vegetation'}
    return class_labels.get(predicted_class, "Unknown")
