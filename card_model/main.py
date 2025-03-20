import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import os
import numpy as np
from tensorflow.keras.layers import Rescaling

# Define paths to your dataset
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'
model_save_path = 'card_model.h5'

# Set image dimensions and batch size
img_height, img_width = 50, 20
batch_size = 8

num_cores = os.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=[-1/20, 1/20],  # Shift by -1 to +1 pixel horizontally
    height_shift_range=[-1/50, 1/50],
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the number of classes
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# Ensure the model output matches the number of classes
if num_classes != 52:
    raise ValueError(f"Expected 52 classes, but found {num_classes} classes in the dataset.")

# Function to save the model
def save_model(model, path):
    model.save(path)
    print(f'Model saved to {path}')

# Function to load the model
def load_model(path):
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        print(f'Model loaded from {path}')
        return model
    else:
        print(f'No model found at {path}. Training a new model...')
        return None

# Load the model if it exists, otherwise train a new one
model = load_model(model_save_path)

if model is None:
    # Define the model with adjusted parameters for 20x50 images
    model = Sequential([
        Rescaling(1./255),
        
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary to understand the dimensions
    model.summary()
    
    # Function to save the best model during training
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=10000,  # Adjust the number of epochs as needed
        callbacks=[checkpoint]
    )

    # Save the trained model
    save_model(model, model_save_path)

# Function to predict a new image
def predict_image(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0]).numpy()
    class_labels = list(train_generator.class_indices.keys())
    return class_labels[predicted_class]

# Example usage
if __name__ == "__main__":
    image_path = 'card_0002_normalized.png'  # Replace with your test image path
    prediction = predict_image(image_path, model)
    print(f'Predicted card: {prediction}')