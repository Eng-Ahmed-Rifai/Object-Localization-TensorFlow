"""
Dual-Head Object Localization Architecture in TensorFlow / Keras.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_localization_model(input_shape=(128, 128, 3), num_classes=9):
    inputs = layers.Input(shape=input_shape)
    
    # Feature extractor backbone
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    
    # Head 1: Class Classification
    c = layers.Dense(128, activation='relu')(x)
    c_out = layers.Dense(num_classes, activation='softmax', name='class_output')(c)
    
    # Head 2: Bounding Box Coordinate Regression (xmin, ymin, xmax, ymax)
    b = layers.Dense(128, activation='relu')(x)
    b_out = layers.Dense(4, activation='sigmoid', name='bbox_output')(b)
    
    model = Model(inputs=inputs, outputs=[c_out, b_out])
    model.compile(
        optimizer='adam',
        loss={'class_output': 'categorical_crossentropy', 'bbox_output': 'mse'},
        metrics={'class_output': 'accuracy', 'bbox_output': 'mse'}
    )
    return model
