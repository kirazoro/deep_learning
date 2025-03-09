#Image based disease diagnosis
#Image Preprocessing → CNN → Fully Connected Layers

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load dataset
train_dir = 'chest_xray/train'
test_dir = 'chest_xray/test'

datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory(train_dir, target_size=(150, 150), class_mode='binary', batch_size=32)
test_data = datagen.flow_from_directory(test_dir, target_size=(150, 150), class_mode='binary', batch_size=32)

# Model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=5, validation_data=test_data)

# Evaluate model
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy:.2f}")
