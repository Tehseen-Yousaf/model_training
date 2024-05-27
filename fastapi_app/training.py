from tensorflow.keras.applications import MobileNetV2, EfficientNetB0  # Adjust model import based on your choice
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
from collections import Counter

train_data_dir ="images/train"
validation_data_dir = "images/validation"
test_data_dir = "images/test"
image_size = (224, 224)
batch_size = 32
epochs = 10

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rotation_range=20)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical")


validation_generator = None
test_generator = None
if validation_data_dir:
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical")


if test_data_dir:
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical")

# Load the pre-trained model, freeze base layers
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(image_size[0], image_size[1], 3))
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation="softmax")(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(train_generator, epochs=epochs, steps_per_epoch=len(train_generator),
                    validation_data=validation_generator if validation_generator else None,
                    verbose=1)


model.save('App_model.h5')