# app.py
import streamlit as st
from pdf_extractor import extract_text_from_pdf
from llm_interface import create_llm_chain, ask_question

# Initialize the LLM chain for question-answering
llm_chain = create_llm_chain()

st.title("PDF Chatbot")
st.write("Upload a PDF and ask questions about its content!")

# File uploader for PDF
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

# Check if a PDF is uploaded
if uploaded_pdf is not None:
    # Extract text from the uploaded PDF
    pdf_content = extract_text_from_pdf(uploaded_pdf)

    # Show input box for user question
    user_question = st.text_input("Ask a question about the document:")
    
    # Display button for asking the question
    if st.button("Ask") and user_question:
        # Generate response
        response = ask_question(llm_chain, pdf_content, user_question)
        st.write(response)
