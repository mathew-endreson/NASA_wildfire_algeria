# predict.py

import tensorflow as tf
import numpy as np
import cv2  # OpenCV
import imageio  # For creating GIFs
import os
import random

def predict_on_image(image_path, model):
    """Loads an image and makes a prediction using the trained model."""
    img = tf.keras.utils.load_img(
        image_path, target_size=(256, 256)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.sigmoid(predictions[0])
    confidence = 100 * score.numpy()[0]
    
    if confidence > 50:
        return 'fire', confidence
    else:
        return 'no_fire', 100 - confidence

def main():
    """Main function to run prediction and create visualization."""
    # --- Load the Model ---
    MODEL_PATH = 'wildfire_detector_algeria.keras'
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please run train_model.py first.")
        return
        
    print("--- Loading trained model ---")
    model = tf.keras.models.load_model(MODEL_PATH)

    # --- Select a random test image from the 'fire' directory ---
    fire_images_dir = 'data/fire'
    if not os.listdir(fire_images_dir):
        print("Error: No images found in data/fire. Please run download_data.py.")
        return
        
    random_image_name = random.choice(os.listdir(fire_images_dir))
    test_image_path = os.path.join(fire_images_dir, random_image_name)
    print(f"--- Predicting on test image: {test_image_path} ---")

    # --- Make Prediction ---
    predicted_class, confidence = predict_on_image(test_image_path, model)
    print(f"Model prediction: '{predicted_class}' with {confidence:.2f}% confidence.")

    # --- Create Visualization GIF ---
    original_image = cv2.imread(test_image_path)
    annotated_image = original_image.copy()

    if predicted_class == 'fire':
        color = (0, 0, 255)  # Red in BGR format
        text = f"FIRE DETECTED ({confidence:.2f}%)"
        cv2.rectangle(annotated_image, (5, 5), (250, 250), color, 4)
        cv2.putText(annotated_image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else: # Should not happen since we picked from the fire folder, but good practice
        color = (0, 255, 0) # Green
        text = f"NO FIRE ({confidence:.2f}%)"
        cv2.putText(annotated_image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Save the annotated image for inspection
    ANNOTATED_IMG_PATH = 'annotated_result.jpg'
    cv2.imwrite(ANNOTATED_IMG_PATH, annotated_image)
    print(f"Annotated image saved to {ANNOTATED_IMG_PATH}")
    
    # Create and save the GIF
    GIF_PATH = 'prediction_algeria.gif'
    print(f"--- Creating GIF: {GIF_PATH} ---")
    with imageio.get_writer(GIF_PATH, mode='I', duration=1.5, loop=0) as writer:
        writer.append_data(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        writer.append_data(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    print("\n--- Process complete! ---")
    print(f"Your final GIF is ready: {GIF_PATH}. Post this on LinkedIn!")

if __name__ == "__main__":
    main()
    