from bs4 import BeautifulSoup
import requests

def scrape_url(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.get_text(separator=" ").strip()
    return " ".join(content.split())
