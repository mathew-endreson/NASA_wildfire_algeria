# train_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def main():
    """Main function to build, train, and save the model."""
    # --- Configuration ---
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 32
    DATA_DIR = 'data'
    EPOCHS = 15 # Increased epochs for better training

    # --- Load and Prepare Datasets ---
    print("--- Loading and preparing datasets ---")
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    print("Class Names:", train_ds.class_names)
    
    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # --- Build the CNN Model ---
    print("--- Building the model ---")
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    # --- Train the Model ---
    print("\n--- Starting model training ---")
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=EPOCHS
    )

    # --- Save the Model ---
    MODEL_PATH = 'wildfire_detector_algeria.keras'
    model.save(MODEL_PATH)
    print(f"\nModel saved successfully to {MODEL_PATH}")

    # --- Visualize and Save Training History ---
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(EPOCHS)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    # Save the plot to a file
    PLOT_PATH = 'training_history.png'
    plt.savefig(PLOT_PATH)
    print(f"Training history plot saved to {PLOT_PATH}")
    # plt.show() # Uncomment if you want a pop-up window with the plot


if __name__ == "__main__":
    main()
    