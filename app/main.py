from fastapi import FastAPI, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uuid
from services.scraper import scrape_url  # Function to scrape content from a URL
from services.pdf_extractor import extract_text_from_pdf  # Function to extract text from a PDF
from services.embedding import add_content, query_content, data_store, clean_text  # Helper functions for embeddings

app = FastAPI()

# In-memory store for content metadata (e.g., chat ID and processed text)
content_store = {}

@app.post("/process_url")
async def process_url(url: str = Form(...)):
    """
    Endpoint to process text content from a URL.
    
    Parameters:
    - url: The URL to scrape and process.

    Steps:
    - Scrape the content from the URL.
    - Clean the scraped text.
    - Generate a unique chat ID.
    - Add the processed content to the embedding system and store metadata.
    """
    # Generate a unique chat ID for the processed content
    chat_id = str(uuid.uuid4())
    
    # Scrape and clean the content
    content = scrape_url(url)
    text = clean_text(content)
    
    # Store the processed content in embeddings and the local store
    add_content(chat_id, text)
    content_store[chat_id] = text
    
    # Return a response with the chat ID
    return JSONResponse({
        "chat_id": chat_id,
        "message": "URL content processed and stored successfully."
    })

@app.post("/process_pdf")
async def process_pdf(file: UploadFile):
    """
    Endpoint to process text content from a PDF file.
    
    Parameters:
    - file: The uploaded PDF file.

    Steps:
    - Validate the file type (must be a PDF).
    - Extract text from the PDF.
    - Clean the extracted text.
    - Generate a unique chat ID.
    - Add the processed content to the embedding system and store metadata.
    """
    # Validate the file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF.")
    try:
        # Extract and clean text from the PDF
        text = extract_text_from_pdf(file)
        text = clean_text(text)
        
        # Generate a unique chat ID
        chat_id = str(uuid.uuid4())
        
        # Store the processed content in embeddings and the local store
        add_content(chat_id, text)
        content_store[chat_id] = {"content": text}
        
        # Return a response with the chat ID
        return {"chat_id": chat_id, "message": "PDF content processed and stored successfully."}
    except Exception as e:
        # Handle exceptions and log errors
        print(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the PDF.")

@app.post("/chat")
async def chat(chat_id: str = Form(...), question: str = Form(...)):
    """
    Endpoint to retrieve answers to a user's query based on stored content.
    
    Parameters:
    - chat_id: The unique identifier for the stored content.
    - question: The user's query.

    Steps:
    - Validate if the chat ID exists in the data store.
    - Use embeddings to find the most relevant section of the content for the query.
    - Return the response.
    """
    # Validate the chat ID
    if chat_id not in data_store:
        raise HTTPException(status_code=404, detail="Chat ID not found.")
    
    # Query the content using the embeddings
    response = query_content(chat_id, question)
    
    # Return the response as JSON
    return JSONResponse({"response": response})

@app.get("/debug/content_store")
async def get_data_store():
    """
    Debugging endpoint to retrieve the in-memory content store.
    
    This endpoint allows inspection of stored content (without exposing embeddings).
    """
    # Optionally, serialize data store to hide embeddings
    # serialized_data_store = {
    #     key: {
    #         "content": value["content"]
    #         # "embeddings": "Embedding data hidden for brevity",  # Hide embeddings if needed
    #     }
    #     for key, value in content_store.items()
    # }
    return JSONResponse(content_store)
