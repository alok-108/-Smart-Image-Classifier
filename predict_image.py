# predict_image.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Load model and class names
model = tf.keras.models.load_model("saved_model/image_classifier")
class_names = model.classes if hasattr(model, 'classes') else ['class_0', 'class_1']

def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]

    print(f"Predicted: {predicted_class} ({100 * np.max(score):.2f}% confidence)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("❌ Please provide an image path.")
