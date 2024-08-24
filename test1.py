import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# Define the class names
classnames = np.array(["cardboard", "glass", "metal", "paper", "plastic", "trash"])

# Function to preprocess the image
def process_image(image_path, IMG_SIZE=224):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

# Load the model only if it hasn't been loaded before
if 'mobile_net_v2_model' not in globals():
    try:
        mobile_net_v2_model = tf.keras.models.load_model("mobile_net_v2_model.h5")
    except Exception as e:
        print("Error loading the model:", e)
        exit()

# Function to make predictions and display the result
def predict_image(img_path):
    try:
        # Preprocess the image
        image = process_image(img_path)
        image = np.expand_dims(image, axis=0)

        # Make predictions using the loaded model
        predictions = mobile_net_v2_model.predict(image)
        predicted_class_index = np.argmax(predictions)
        predicted_class = classnames[predicted_class_index]

        # Display the image with the predicted class label
        img = plt.imread(img_path)
        plt.title(f"Prediction: {predicted_class}")
        plt.imshow(img)
        plt.show()
    except Exception as e:
        print("Error processing the image:", e)

# Main loop
while True:
    # Get the image path from the user
    img_path = input("Enter the image path for prediction (or type 'exit' to quit): ")

    if img_path.lower() == 'exit':
        print("Exiting the program.")
        break

    # Predict and display the result
    predict_image(img_path)
