# AI-Powered Content Processor and Chat Service

## Overview

This project implements an AI-powered content query system that uses embeddings and a multi-agent architecture to process and query content from various sources such as URLs and PDF files. It utilizes the **Sentence Transformers** model to create embeddings, **FastAPI** for web service endpoints, and integrates various components for scraping, PDF extraction, and querying.

The system allows users to:
- Extract and store content from URLs and PDFs.
- Generate and store content embeddings.
- Query content based on user questions, retrieving the most relevant information.

## Features

- **Content Processing:** Extracts and processes content from URLs and PDF files.
- **Embedding Storage:** Uses Sentence Transformers to create and store content embeddings.
- **Query System:** Leverages cosine similarity to return the most relevant content for a given query.
- **Multi-Agent Architecture:** Allows the flexible addition of content from different sources into the data store.

## Technologies Used

- **FastAPI:** For building the RESTful web service.
- **Sentence-Transformers:** For generating content embeddings.
- **Sklearn:** For calculating cosine similarity between the query and stored embeddings.
- **BeautifulSoup:** For scraping content from URLs.
- **PyPDF2:** For extracting text from PDF files.
- **UUID:** For generating unique chat IDs to manage content and queries.
- **Serper API (Optional):** For enhanced web scraping capabilities (if included).

### Installation

### Prerequisites
- Python 3.10+
- Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```
