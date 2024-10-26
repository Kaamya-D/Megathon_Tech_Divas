from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini Pro model with an empty history
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question, chat_history):
    """
    Function to get the response from Gemini Pro with full conversation context.
    """
    # Add the full chat history into the current question to provide context
    conversation = " ".join([f"{role}: {text}" for role, text in chat_history])
    response = chat.send_message(f"{conversation}\nUser: {question}", stream=True)
    return response

# Initialize the Streamlit app
st.set_page_config(page_title="Mental Health Concern Classification")

st.header("Gemini LLM Chatbot with Context")

# Initialize chat history in session state if not already initialized
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []  # List to store the chat history

# Input text area for user input
input_text = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit and input_text:
    # Get the Gemini response with full context (including prior messages)
    response = get_gemini_response(input_text, st.session_state['chat_history'])
    
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