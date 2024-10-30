# pdf_extractor.py
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file):
    """Extracts text from each page of the PDF."""
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text
