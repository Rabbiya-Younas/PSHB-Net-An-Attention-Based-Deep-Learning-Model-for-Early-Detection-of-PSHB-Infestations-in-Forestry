# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 18:51:10 2025

@author: Rabbiya Younas
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 18:46:26 2025

@author: raza
"""

# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import seaborn as sns

# Set seed
tf.random.set_seed(42)

# Paths and Parameters
data_dir = "E:/private/PSHB images  2 (1)/dataset"  # <-- Update this path
img_size = (224, 224)
batch_size = 32

# Image augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='training')
val_data = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='validation', shuffle=False)

from tensorflow.keras import layers, models
import tensorflow as tf

# Enhanced Proposed CNN Model with Fourier + SE (Squeeze-and-Excitation) blocks
def build_custom_model(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)

    # Initial Conv Layer
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Fourier Layer
    def fourier_layer(x):
        x_complex = tf.cast(x, tf.complex64)
        fft = tf.signal.fft2d(x_complex)
        return tf.math.abs(fft)

    x = layers.Lambda(fourier_layer)(x)

    # Conv Block + SE Layer
    def conv_se_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Squeeze-Excitation
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(filters // 16, activation='relu')(se)
        se = layers.Dense(filters, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, filters))(se)
        x = layers.Multiply()([x, se])

        return layers.MaxPooling2D((2, 2))(x)

    x = conv_se_block(x, 64)
    x = conv_se_block(x, 128)
    x = conv_se_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model


# Transfer learning
def build_transfer_model(base_class, preprocess_func):
    base = base_class(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    inputs = layers.Input(shape=(224, 224, 3))
    x = preprocess_func(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs)

# Compile and Train
model_dict = {
    "Custom_CNN": build_custom_model(),
    "MobileNetV2": build_transfer_model(MobileNetV2, mobilenet_preprocess),
    "ResNet50": build_transfer_model(ResNet50, resnet_preprocess)
}

histories = {}
metrics_results = []

for name, model in model_dict.items():
    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    es = callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    print(f"\nTraining {name}...")
    history = model.fit(train_data, validation_data=val_data, epochs=30, callbacks=[es])
    histories[name] = history.history

    val_data.reset()
    y_true = val_data.classes
    y_prob = model.predict(val_data)
    y_pred = (y_prob > 0.5).astype('int32')

    metrics_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_prob)
    })

    # Plot predictions
    sample_images, _ = next(val_data)
    preds = model.predict(sample_images[:2])
    for i in range(2):
        plt.imshow(sample_images[i])
        label = 'Infected' if preds[i] > 0.5 else 'Non-Infected'
        conf = preds[i][0] if preds[i] > 0.5 else 1 - preds[i][0]
        plt.title(f"{label} (Confidence: {conf:.2f})")
        plt.axis('off')
        plt.show()

# Plot training curves
for name, hist in histories.items():
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist['accuracy'], label='Train Acc')
    plt.plot(hist['val_accuracy'], label='Val Acc')
    plt.title(f'{name} - Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist['loss'], label='Train Loss')
    plt.plot(hist['val_loss'], label='Val Loss')
    plt.title(f'{name} - Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Print results
results_df = pd.DataFrame(metrics_results)
print("\nEvaluation Metrics Summary:")
print(results_df)
