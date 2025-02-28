import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample dataset
reviews = ["Great product!", "Worst purchase ever.", "I love this phone!", "Terrible quality.", "Highly recommended!"]
labels = [1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative

# Tokenization and Padding
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')

# Build a simple Dense Neural Network
model = keras.Sequential([
    keras.layers.Embedding(1000, 16, input_length=10),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, np.array(labels), epochs=10)

# Evaluate model
loss, accuracy = model.evaluate(padded_sequences, np.array(labels))
print(f"Model Accuracy: {accuracy:.2f}")
