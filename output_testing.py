import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# --- 1. Define Constants and Load Model ---

# Constants must match the training setup
IMAGE_SIZE = (150, 150)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Define the path to your saved model (replace if you saved it differently)
MODEL_PATH = 'brain_tumor_vgg16_model.h5' 
# Assuming you save the model after training: model.save(MODEL_PATH)

try:
    # Load the trained model
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model successfully loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you saved the model after training.")
    # Exit if model fails to load

# --- 2. Function to Load, Preprocess, and Predict ---

def predict_single_image(img_path, model):
    """Loads, preprocesses, and makes a prediction on a single image."""
    
    # Load the image
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    
    # Display the image for verification
    plt.imshow(img)
    plt.title(f"Input Image: {img_path.split('/')[-1]}")
    plt.axis('off')
    plt.show()

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to create a batch size of 1 (required by Keras)
    # Shape changes from (150, 150, 3) to (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Rescale the pixels (MUST match the 1./255 rescaling used during training!)
    # VGG16 weights were trained on ImageNet with a specific preprocessing, 
    # but since you only used 1./255 scaling, we stick to that.
    img_array = img_array / 255.0 

    # Make the prediction
    # model.predict returns an array of probabilities (one for each class)
    predictions = model.predict(img_array)
    
    # Get the predicted class index (highest probability)
    predicted_class_index = np.argmax(predictions[0])
    
    # Get the corresponding class name
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    
    # Get the confidence score
    confidence = predictions[0][predicted_class_index]

    # --- Print Results ---
    print("\n--- Prediction Results ---")
    print(f"Predicted Class: {predicted_class_name}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    if predicted_class_name == 'notumor':
        print("\n✅ Diagnosis: No Tumor Detected.")
    else:
        print(f"\n⚠️ Diagnosis: Brain Tumor Detected ({predicted_class_name}).")
        
    # Optional: print all probabilities
    # print("\nAll Probabilities:", dict(zip(CLASS_NAMES, predictions[0])))


# --- 3. Execute the Prediction ---

# TODO: Replace this with the actual path to your test image file
NEW_IMAGE_PATH = '/path/to/your/new_mri_scan.jpg' 

if 'loaded_model' in locals():
    predict_single_image(NEW_IMAGE_PATH, loaded_model)
