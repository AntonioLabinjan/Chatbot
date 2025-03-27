import streamlit as st
import pdfplumber
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import asyncio

# Define an optimized prompt template
template = """ 
You are an AI assistant that answers questions based on the provided conversation history and document contents.

## Document Content:
{context}

## Previous Conversation:
{history}

## Question:
{question}

## Answer:
"""

# Initialize the model
model = OllamaLLM(model='llama3')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Streamlit UI
st.title("AI ChatBot with PDF Support 📄🤖")
st.write("Upload a PDF and ask questions about it!")

# Session state for context and history
if "history" not in st.session_state:
    st.session_state.history = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# Upload and process PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

        # Store summarized text in session state
        st.session_state.pdf_text = text[:3000]  # Limit to avoid overload
        st.success("PDF uploaded successfully! You can now ask questions.")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a question about the document...")
if user_input:
    with st.spinner("Thinking..."):
        st.session_state.history.append({"role": "user", "content": user_input})

        async def get_response():
            response_stream = chain.stream({
                "context": st.session_state.pdf_text,  # Provide only relevant document content
                "history": "\n".join([msg["content"] for msg in st.session_state.history[-5:]]),  # Limit history
                "question": user_input
            })
            
            response = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
                for chunk in response_stream:
                    response += chunk
                    placeholder.markdown(response)

            return response

        # Run async processing
        response_text = asyncio.run(get_response())

        # Append AI response to chat history
        st.session_state.history.append({"role": "assistant", "content": response_text})
