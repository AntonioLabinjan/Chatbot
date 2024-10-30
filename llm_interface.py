import os
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face API key from environment variables
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
if not huggingface_api_key:
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables.")

def create_llm_chain():
    """Initializes the Hugging Face pipeline with an advanced question-answering model."""
    # Initialize Hugging Face pipeline for question-answering with Flan-T5
    qa_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",  # For better results, you could use flan-t5-xl or flan-t5-xxl if resources allow
        #use_auth_token=huggingface_api_key
    )
    return qa_pipeline

def ask_question(qa_pipeline, pdf_content, question):
    """Asks a question to the model using the extracted PDF content as context."""
    # Format the input for Flan-T5
    input_text = f"Context: {pdf_content}\nQuestion: {question}\nAnswer:"
    
    # Run the pipeline with the given inputs
    response = qa_pipeline(input_text, max_length=256, truncation=True)
    return response[0]['generated_text']
