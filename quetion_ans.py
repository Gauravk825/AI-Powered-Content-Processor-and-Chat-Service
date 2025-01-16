from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# Load SentenceTransformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load LLM model
tokenizer = AutoTokenizer.from_pretrained("gpt2") 
llm_model = AutoModelForCausalLM.from_pretrained("gpt2") 

# Sample stored content (replace with your actual content)
stored_content = """
This document discusses the importance of artificial intelligence in modern society. 
It covers topics such as machine learning, deep learning, and natural language processing. 
The document also explores the potential impact of AI on various industries, including healthcare, finance, and transportation.
"""

def generate_response_with_llm(question, stored_content):
    """
    Generates a response using an LLM.

    Args:
        question: The user's question.
        stored_content: The stored document content.

    Returns:
        The LLM-generated response.
    """
    prompt = f"Question: {question}\n\nContext: {stored_content}\n\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = llm_model.generate(input_ids, max_length=100, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]

def handle_chat_request(request_body):
    """
    Handles the chat request and returns the most relevant response.

    Args:
        request_body: JSON object containing "chat_id" and "question"

    Returns:
        JSON object containing the "response"
    """

    try:
        chat_id = request_body["chat_id"]
        question = request_body["question"]

        # Generate embeddings for the question and stored content
        question_embedding = sentence_model.encode(question)
        content_embedding = sentence_model.encode(stored_content)

        # Calculate cosine similarity between the embeddings
        cosine_scores = util.cos_sim(question_embedding, content_embedding)

        # Find the index of the most relevant section (in this case, the entire document) 
        # (Since we have only one section, this is always 0)
        most_relevant_index = cosine_scores.argmax() 

        # Generate response using LLM
        response = generate_response_with_llm(question, stored_content)

        #return {"response": response}
        return most_relevant_index

    except KeyError as e:
        return {"error": f"Missing field in request: {e}"}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Example usage
request_body = {
    "chat_id": "unique_chat_1",
    "question": "What is the main idea of the document?"
}

response = handle_chat_request(request_body)
#print(json.dumps(response, indent=4))
print(response)