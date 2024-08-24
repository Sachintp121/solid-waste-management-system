import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import preprocess_input
from ubidots import ApiClient

# Function to predict waste type and update Ubidots variable
def predict_and_update(image_path):
    # Load the pre-trained model
    model = load_model("garbage_classifier.vgg_19.h5")
    class_names = ['battery', 'glass', 'metal', 'organic', 'paper', 'plastic']

    # Load the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Predict waste type
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]

    # Display the result
    print("Predicted waste type:", predicted_class)

    # Update Ubidots variable based on predicted class
    if predicted_class == "plastic":
        Variable.save_value({'value': 1})
    elif predicted_class == "paper":
        Variable.save_value({'value': 2})
    elif predicted_class == "trash":
        Variable.save_value({'value': 0})
    elif predicted_class == "battery":
        Variable.save_value({'value': 0})
    elif predicted_class == "metal":
        Variable.save_value({'value': 4})
    elif predicted_class == "glass":
        Variable.save_value({'value': 5})
    elif predicted_class == "cardboard":
        Variable.save_value({'value': 0})

    # Display selected image with prediction
    img = Image.open(image_path)
    img = img.resize((400, 400))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img
    prediction_label.config(text=f"Predicted waste type: {predicted_class}")

    # Display finishing with exit code
    print("Finishing with exit code 0")

# Function to select image
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        predict_and_update(file_path)

# Create Tkinter window
root = tk.Tk()
root.title("Waste Type Classifier")

# Ubidots setup
api = ApiClient(token="BBUS-bMlMkhXqNzssxcuyxwN47kadRu0gOq")
Variable = api.get_variable("663ce3d6f03b66308887879d")

# Create label to display image
image_label = tk.Label(root)
image_label.pack()

# Create label to display prediction
prediction_label = tk.Label(root, text="", font=("Arial", 16))
prediction_label.pack()

# Create button to select image
select_button = tk.Button(root, text="Select Image", command=select_image, bg="red", fg="blue", font=("Arial", 16))
select_button.pack(side=tk.BOTTOM, pady=10)

# Project title and description
title_label = tk.Label(root, text="Waste Classification Using CNN and Hardwares", bg="sky blue", fg="black", font=("Arial", 20, "bold"))
title_label.pack(pady=10)

description_label = tk.Label(root, text="A Project on Waste Classification and Management using Convolutional Neural Networks", bg="light yellow", fg="black", font=("Arial", 16))
description_label.pack(pady=10)

# Presented by names with USN
presented_by_label = tk.Label(root, text="Presented by:", bg="green", fg="black", font=("Arial", 18))
presented_by_label.pack(pady=5)

name1_label = tk.Label(root, text="Sachin (USN: 4HG20CS021)", bg="green", fg="black", font=("Arial", 18))
name1_label.pack(pady=5)

name2_label = tk.Label(root, text="Rohith (USN: 4HG20CS019)", bg="green", fg="black", font=("Arial", 18))
name2_label.pack(pady=5)

name3_label = tk.Label(root, text="Karthik (USN: 4HG20CS007)", bg="green", fg="black", font=("Arial", 18))
name3_label.pack(pady=5)

name4_label = tk.Label(root, text="Nirmitha (USN: 4HG21CS419)", bg="green", fg="black", font=("Arial", 18))
name4_label.pack(pady=5)

name5_label = tk.Label(root, text="Bhoomika (USN: 4HG21CS403)", bg="green", fg="black", font=("Arial", 18))
name5_label.pack(pady=5)

# Run Tkinter event loop
root.mainloop()
