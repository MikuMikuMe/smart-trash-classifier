Creating a smart trash classifier involves setting up an AI model to categorize waste into predefined categories such as plastic, metal, paper, glass, etc. Below is a simplified Python program using a convolutional neural network (CNN) model with the TensorFlow and Keras libraries to classify waste images. The model assumes that you have a dataset ready to be used for training and testing.

```python
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def create_data_generators(train_dir, validation_dir, image_size=(150, 150), batch_size=32):
    """
    Create training and validation data generators with data augmentation for the training set.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator

def build_model(input_shape, num_classes):
    """
    Build a convolutional neural network model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def main():
    # Path to the training and validation directories
    train_dir = 'path/to/train_dir'
    validation_dir = 'path/to/validation_dir'

    # Ensure directories exist
    if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
        raise FileNotFoundError("Training or validation directory not found")

    # Parameters
    image_size = (150, 150)
    batch_size = 32
    num_classes = 4  # Example: plastic, metal, paper, glass
    input_shape = (150, 150, 3)

    # Create data generators
    train_generator, validation_generator = create_data_generators(train_dir, validation_dir, image_size, batch_size)

    # Build and compile model
    model = build_model(input_shape, num_classes)

    # Callback for early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=50,
            callbacks=[early_stopping]
        )
    except Exception as e:
        print(f"Error during model training: {str(e)}")

    # Save the model
    model_path = 'smart_trash_classifier.h5'
    try:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error while saving the model: {str(e)}")

if __name__ == '__main__':
    main()
```

### Key Components:

1. **Data Preparation**: 
   - Use `ImageDataGenerator` for data loading and augmentation. Adjust `train_dir` and `validation_dir` paths for your dataset locations.

2. **Model Architecture**: 
   - A simple CNN model with three convolutional layers, pooling layers, and dropout for regularization.

3. **Training & Validation**:
   - Includes early stopping to reduce overfitting by monitoring validation loss.

4. **Error Handling**: 
   - Checks for directory existence and handles errors during model training and saving.

### Instructions:
- You must have a pre-sorted dataset of images, separated into directories by category.
- Make sure to adjust the number of classes and the dataset path according to your needs.
- This is a basic template. Real-world applications may require more advanced models with hyperparameter tuning and possibly transfer learning techniques.