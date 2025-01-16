from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Initialize the SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Data store to hold content and its corresponding embeddings
data_store = {}

def clean_text(text):
    """
    Cleans the input text by:
    - Removing extra spaces, tabs, and line breaks
    - Removing special characters (e.g., punctuation)
    """
    text = re.sub(r'[^\w\s]', '', text)  # Remove all characters except alphanumerics and spaces
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def add_content(content_id: str, content: str):
    """
    Adds content to the `data_store` and computes embeddings for the content.
    
    Parameters:
    - content_id: A unique identifier for the content
    - content: The text content to be stored and analyzed
    """
    # Compute embeddings for the entire content as a single vector
    embeddings = model.encode(content, convert_to_tensor=False).tolist()  # Convert to list
    # Store the content and its embeddings in the data store
    data_store[content_id] = {"content": content, "embeddings": embeddings}

def query_content(content_id: str, query: str) -> str:
    """
    Finds the most relevant paragraph in the stored content for a given query.

    Parameters:
    - content_id: The unique identifier for the content to search in
    - query: The user query for which a relevant paragraph is sought

    Returns:
    - The paragraph most relevant to the query
    """
    # Check if the content ID exists in the data store
    if content_id not in data_store:
        raise ValueError("Content ID not found in the data store.")
    
    # Retrieve stored embeddings and content
    stored_embeddings = np.array(data_store[content_id]["embeddings"])  # Ensure it's a NumPy array
    stored_content = data_store[content_id]["content"]

    # Split content into paragraphs (assuming paragraphs are separated by newlines)
    paragraphs = stored_content.split("\n")
    # Compute embeddings for each paragraph
    paragraph_embeddings = model.encode(paragraphs, convert_to_tensor=False)

    # Encode the query into an embedding
    query_embedding = model.encode(query, convert_to_tensor=False)
    # Compute cosine similarity between the query embedding and paragraph embeddings
    similarities = cosine_similarity([query_embedding], paragraph_embeddings)

    # Find the index of the most relevant paragraph (highest similarity score)
    most_relevant_index = np.argmax(similarities)
    # Retrieve the most relevant paragraph
    most_relevant_paragraph = paragraphs[most_relevant_index]
    
    # Return the most relevant paragraph as the answer
    return most_relevant_paragraph
