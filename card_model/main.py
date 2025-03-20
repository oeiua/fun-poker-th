import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

# Optimize CPU performance
num_cores = os.cpu_count()
print(f"Number of CPUs: {num_cores}")

# Configure TensorFlow for optimal CPU performance
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(8)  # Usually best at 2-4 for intra-thread coordination

# Enhanced CPU optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable Intel MKL-DNN optimizations if available
os.environ['OMP_NUM_THREADS'] = str(num_cores)
os.environ['KMP_BLOCKTIME'] = '0'  # No waiting between parallel regions
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'  # Core affinity optimization
os.environ['TF_CPU_ALGORITHM'] = '0'  # Use specified algorithm for CPU operations

# Try to enable additional optimizations if available
try:
    # Enable operation fusion optimization
    tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra)
    print("XLA optimization enabled")
except:
    print("XLA optimization not available in this TensorFlow version")

# Try to set memory growth options
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('CPU')[0], True)
    print("Memory growth enabled")
except:
    print("Memory growth option not available for CPU")

# Define paths to your dataset
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'
model_save_path = 'card_model.h5'

# Set image dimensions and increase batch size for better CPU utilization
img_height, img_width = 50, 20
batch_size = 16  # Increased from 8 to better utilize CPU cores

def aspect_ratio_transformer(image):
    """
    Optimized version of aspect ratio transformer using numpy operations
    where possible to reduce overhead.
    """
    # Convert to PIL Image for manipulation (assumes image is in [0,1] range from rescale)
    img_array = (image * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Original dimensions
    width, height = img.size

    img_resized = img.resize((height, width), Image.LANCZOS)

    # Convert back to numpy array and rescale to [0, 1]
    return np.array(img_resized) / 255.0

# Data prefetching and buffering parameters
AUTOTUNE = tf.data.AUTOTUNE

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    width_shift_range=[-1/20, 1/20],  
    height_shift_range=[-1/50, 1/50],
    fill_mode='nearest',
    # preprocessing_function=aspect_ratio_transformer
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
    # Using more efficient model architecture for CPU
    model = Sequential([
        # No need for explicit rescaling layer as ImageDataGenerator already does this
        
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),  # Add dropout for regularization
        Dense(num_classes, activation='softmax')
    ])

    # Use a lower learning rate for more stable CPU training
    optimizer = Adam(learning_rate=0.01)
    
    # Compile the model
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    # Print the model summary to understand the dimensions
    model.summary()
    
    # Callbacks for better training
    callbacks = [
        # Save best model
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Learning rate schedule
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=100,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Calculate steps more reliably
    steps_per_epoch = max(1, train_generator.samples // batch_size)
    validation_steps = max(1, validation_generator.samples // batch_size)

    # Train the model with a fixed number of epochs
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=100000,  # Increased epochs to allow proper learning
        callbacks=callbacks,
        verbose=1  # Show progress bar
    )

    # Save the trained model
    save_model(model, model_save_path)

# Function to predict a new image - optimized for CPU
def predict_image(image_path, model):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    
    # Run prediction
    predictions = model.predict(img_array, verbose=0)  # Disable verbose output
    predicted_class = tf.argmax(predictions[0]).numpy()
    class_labels = list(train_generator.class_indices.keys())
    return class_labels[predicted_class]

# Example usage
if __name__ == "__main__":
    image_path = 'card_0001_normalized.png'  # Replace with your test image path
    prediction = predict_image(image_path, model)
    print(f'Predicted card: {prediction}')