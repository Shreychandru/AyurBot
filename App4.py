import streamlit as st
import nltk
import random
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

# Download NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Load data from CSV
data = pd.read_csv('Formulation-Indications.csv')

# Preprocess data
lemmatizer = WordNetLemmatizer()

patterns = []
tags = []
word_to_int = {}
for index, row in data.iterrows():
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(row['Reference text'])]
    patterns.append(pattern_words)
    tags.append(row['Class'])
    for word in pattern_words:
        if word not in word_to_int:
            word_to_int[word] = len(word_to_int) + 1

# Encode patterns with integer values
encoded_patterns = []
for pattern in patterns:
    encoded_pattern = [word_to_int[word] for word in pattern]
    encoded_patterns.append(encoded_pattern)

# Pad sequences to equal length
max_len = max(len(p) for p in encoded_patterns)
padded_patterns = pad_sequences(encoded_patterns, maxlen=max_len, padding='post')

# Create a dictionary to map intent names to numerical values
tag_to_int = {}
int_to_tag = {}
for i, tag in enumerate(set(tags)):
    tag_to_int[tag] = i
    int_to_tag[i] = tag

# Encode tags using the dictionary
encoded_tags = [tag_to_int[tag] for tag in tags]
encoded_tags = to_categorical(encoded_tags, num_classes=len(tag_to_int))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_patterns, encoded_tags, test_size=0.2, random_state=42)

# Define the RNN model architecture
model = Sequential([
    Embedding(input_dim=len(word_to_int) + 1, output_dim=128),
    LSTM(128),
    Dense(len(tag_to_int), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the RNN model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

def chatbot_response(text, data):
    pattern = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(text)]
    encoded_pattern = [word_to_int.get(word, 0) for word in pattern]
    encoded_pattern = pad_sequences([encoded_pattern], maxlen=max_len, padding='post')
    prediction = model.predict(encoded_pattern)
    predicted_class_index = np.argmax(prediction)
    intent_name = int_to_tag[predicted_class_index]
    response = data[data['Class'] == intent_name]['Name of Medicine'].values
    return response[0] if len(response) > 0 else "I'm sorry, I couldn't understand that."

# Streamlit UI
st.title("Chatbot")

user_input = st.text_input("You: ")

if st.button("Send"):
    output = chatbot_response(user_input, data)
    st.text_area("Bot:", value=output, height=100, max_chars=None, key="output")
