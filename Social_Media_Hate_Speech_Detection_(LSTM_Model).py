import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample dataset
comments = ["I hate this!", "This is amazing.", "You are terrible!", "Great work!", "This is offensive."]
labels = [1, 0, 1, 0, 1]  # 1 = Hate speech, 0 = Normal

# Tokenization and Padding
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')

# Build LSTM-based model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=10),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, np.array(labels), epochs=10)

# Evaluate model
loss, accuracy = model.evaluate(padded_sequences, np.array(labels))
print(f"Model Accuracy: {accuracy:.2f}")
