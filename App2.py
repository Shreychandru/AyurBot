import streamlit as st
import nltk
import random
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np  # Import NumPy for array operations

# Download NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Load intents data
with open('data.json') as file:
    intents = json.load(file)

# Preprocess patterns and tags
patterns = []
tags = []
lemmatizer = WordNetLemmatizer()

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern.split()]
        patterns.append(pattern_words)
        tags.append(intent["tag"])

# Create word vocabulary
word_to_int = {}
for pattern in patterns:
  for word in pattern:
    if word not in word_to_int:
      word_to_int[word] = len(word_to_int) + 1

# Encode patterns with integer values
encoded_patterns = []
for pattern in patterns:
  encoded_pattern = [word_to_int[word] for word in pattern]
  encoded_patterns.append(encoded_pattern)

# Pad sequences to equal length
max_len = max(len(p) for p in encoded_patterns)  # Calculate max_len here
padded_patterns = pad_sequences(encoded_patterns, maxlen=max_len, padding='post')

# Create a dictionary to map intent names to numerical values
tag_to_int = {}
int_to_tag = {}  # Optional for decoding tags if needed later

for i, intent in enumerate(intents["intents"]):
  tag_to_int[intent["tag"]] = i
  int_to_tag[i] = intent["tag"]  # Optional for decoding

# Encode tags using the dictionary
encoded_tags = [tag_to_int[tag] for tag in tags]

# One-hot encode the tags
encoded_tags = to_categorical(encoded_tags, num_classes=len(tag_to_int))

# Define the RNN model architecture
inputs = Input(shape=(max_len,))  # Define the input layer with shape

model = Sequential([Embedding(input_dim=len(word_to_int) + 1, output_dim=128, input_length=max_len),
                    LSTM(128),
                    Dense(len(tag_to_int), activation='softmax')])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ensure data types are NumPy arrays
padded_patterns = np.array(padded_patterns)
encoded_tags = np.array(encoded_tags)

# Print shapes for debugging (optional)
print(f"Shape of padded_patterns: {padded_patterns.shape}")
print(f"Shape of encoded_tags: {encoded_tags.shape}")

# Train the RNN model (replace 1000 with a suitable number of epochs for training)
model.fit(padded_patterns, encoded_tags, epochs=10)

def chatbot_response(text, intents_data):
    pattern = [lemmatizer.lemmatize(word.lower()) for word in text.split()]
    encoded_pattern = [word_to_int.get(word, 0) for word in pattern]
    encoded_pattern = pad_sequences([encoded_pattern], maxlen=max_len, padding='post')
    prediction = model.predict(encoded_pattern)
    predicted_class_index = np.argmax(prediction)
    intent_name = int_to_tag[predicted_class_index]
    for intent in intents_data["intents"]:
        if intent["tag"] == intent_name:
            response = random.choice(intent['responses'])
            return response


# Streamlit UI
st.title("Chatbot")

user_input = st.text_input("You: ")

if st.button("Send"):
    output = chatbot_response(user_input, intents)
    st.text_area("Bot:", value=output, height=100, max_chars=None, key="output")
