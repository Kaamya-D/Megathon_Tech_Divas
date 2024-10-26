from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
load_dotenv()

# Configure the Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the mental health dataset
data = pd.read_csv("/home/user/megathon/mental_health_dataset - Sheet1.csv")  # Replace with your actual file path

# Initialize the Gemini Pro model with an empty history
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question, chat_history, dataset_info=None):
    """
    Function to get the response from Gemini Pro with full conversation context and optional dataset info.
    """
    # Add the full chat history into the current question to provide context
    conversation = " ".join([f"{role}: {text}" for role, text in chat_history])
    message = f"{conversation}\nUser: {question}\n"

    if dataset_info:
        message += (
            f"Here are some relevant details based on your input:\n{dataset_info}\n"
            "Can you provide insights or advice regarding the user's concern?"
        )
    else:
        message += "Can you provide insights or advice regarding the user's concern based solely on your understanding?"

    response = chat.send_message(message, stream=True)
    return response

def analyze_user_input(user_input):
    """
    Analyze user input based on the custom mental health dataset.
    Returns dataset info if a match is found, otherwise returns None.
    """
    # Use TF-IDF to find similarity between user input and dataset entries
    tfidf_vectorizer = TfidfVectorizer()
    
    # Combine the existing 'User Input' with the new user input
    combined_inputs = pd.concat([data['User Input'], pd.Series([user_input])], ignore_index=True)
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_inputs)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Find the most similar entry
    most_similar_idx = cosine_similarities.argmax()
    most_similar_score = cosine_similarities[most_similar_idx]

    # Set a threshold for similarity (you can adjust this value)
    threshold = 0.1  # Adjust this threshold as necessary

    if most_similar_score < threshold:
        return None  # No relevant match found
    
    matching_row = data.iloc[most_similar_idx]

    # Return the extracted details from the dataset as a formatted string
    dataset_info = (
        f"Polarity: {matching_row['Polarity']}, "
        f"Extracted Concern: {matching_row['Extracted Concern']}, "
        f"Category: {matching_row['Category']}, "
        f"Intensity: {matching_row['Intensity']}"
    )
    return dataset_info

# Initialize the Streamlit app
st.set_page_config(page_title="Mental Health Concern Classification")

st.header("Gemini LLM Chatbot with Context and Dataset Analysis")

# Initialize chat history in session state if not already initialized
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []  # List to store the chat history

# Input text area for user input
input_text = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit and input_text:
    # Analyze user input based on the dataset and get the formatted dataset info
    dataset_info = analyze_user_input(input_text)

    # Get the Gemini response with full context (including prior messages and dataset information)
    response = get_gemini_response(input_text, st.session_state['chat_history'], dataset_info)
    
    # Append the user's input and the Gemini's response to the chat history
    st.session_state['chat_history'].append(("You", input_text))  # Add user input

    st.subheader("The Response is:")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot", chunk.text))  # Add bot response

# Display the chat history so far
st.subheader("The Chat History is:")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
