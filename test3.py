import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import preprocess_input

# Load the pre-trained model
model = load_model("garbage_classifier.vgg_19.h5")
class_names = ['battery', 'glass', 'metal', 'organic', 'paper', 'plastic']  # Update with your class names

# Access the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (224, 224))
    img = image.img_to_array(resized_frame)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Predict waste type
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]

    # Overlay predicted waste type on frame
    cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the annotated frame
    cv2.imshow('Waste Type Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
