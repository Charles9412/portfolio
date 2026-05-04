# train_model.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# Define paths
data_dir = "data/processed"
sober_dir = os.path.join(data_dir, "sober")
drunk_dir = os.path.join(data_dir, "drunk")

# Parameters
img_size = (224, 224)
batch_size = 16
epochs = 20

# Data augmentation and loading
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

# Load base model
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])

# Checkpoint callback
checkpoint_path = "models/saved_model/mobilenetv2_best.h5"
checkpoint = ModelCheckpoint(
    checkpoint_path, monitor="val_accuracy", save_best_only=False, verbose=1)

# Train
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint]
)

print("Training complete. Best model saved to:", checkpoint_path)