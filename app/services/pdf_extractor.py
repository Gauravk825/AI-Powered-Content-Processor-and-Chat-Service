from PyPDF2 import PdfReader
from fastapi import UploadFile

def extract_text_from_pdf(pdf_file: UploadFile):
    try:
        # Save the uploaded file temporarily
        temp_file_path = pdf_file.filename
        with open(temp_file_path, "wb") as f:
            f.write(pdf_file.file.read())

        # Read the PDF
        reader = PdfReader(temp_file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        return text.strip()
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise ValueError("Failed to process PDF document.")
