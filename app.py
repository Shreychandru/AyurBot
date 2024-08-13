import streamlit as st
import nltk
import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

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
        patterns.append(" ".join(pattern_words))
        tags.append(intent["tag"])

# Vectorize patterns
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(patterns)

# Encode tags
encoder = LabelEncoder()
encoded_tags = encoder.fit_transform(tags)

# Train the Logistic Regression model
classifier = LogisticRegression(random_state=0, max_iter=10000)
classifier.fit(x, encoded_tags)

def chatbot_response(text, intents_data):
    input_text = vectorizer.transform([text])
    tag = classifier.predict(input_text)[0]
    for intent in intents_data["intents"]:
        if intent["tag"] == encoder.inverse_transform([tag])[0]:
            response = random.choice(intent['responses'])
            return response

# Streamlit UI
st.title("Chatbot")

user_input = st.text_input("You: ")

if st.button("Send"):
    output = chatbot_response(user_input, intents)
    st.text_area("Bot:", value=output, height=100, max_chars=None, key=None)
