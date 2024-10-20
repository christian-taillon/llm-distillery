import os
import requests
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables
MODEL = "claude-3-sonnet-20240229"
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
API_ENDPOINT = 'https://api.anthropic.com/v1/messages'

# OpenWebUI / OPEN AI API Standard
# MODEL = "llama3:405b"
# OPENWEBUI_KEY = os.environ.get('OPENWEBUI_KEY')
# API_ENDPOINT = 'http://openapi.example/api/chat/completions'

def anthropic_api_request(prompt):
    """Make a request to the Anthropic API"""
    logger.info(f"Making API request to Anthropic model: {MODEL}")
    headers = {
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json'
    }
    data = {
        'model': MODEL,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': 1000
    }
    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        logger.debug(f"API Response: {response.text}")
        logger.info("API request completed")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        logger.error(f"Response content: {response.text if response else 'No response'}")
        raise

def generate_topic_title(topic_prompt):
    prompt = f"Generate a concise topic title (2-4 words) based on the following description: {topic_prompt}"
    response = anthropic_api_request(prompt)
    return response['content'][0]['text'].strip()

def generate_document_titles(topic_title):
    prompt = f"""
    Based on the topic "{topic_title}", generate a list of 5-10 document titles that would be valuable for an LLM to reference in a Retrieval-Augmented Generation (RAG) system. These titles should cover various aspects and subtopics related to the main topic. 
    Provide only the titles, one per line, without numbering or additional explanation.
    """
    response = anthropic_api_request(prompt)
    titles = response['content'][0]['text'].strip().split('\n')
    return [title.strip() for title in titles if title.strip()]  # Remove empty titles

def generate_document_content(title):
    prompt = f"""
    Create a comprehensive document on the topic: "{title}"
    This document will be used as a reference for an LLM in a Retrieval-Augmented Generation (RAG) system. 
    Include relevant information, key concepts, examples, and any other pertinent details that would be valuable for an AI to understand and utilize this topic.
    The document should be well-structured, informative, and approximately 500-1000 words in length.
    """
    response = anthropic_api_request(prompt)
    return response['content'][0]['text'].strip()

def main():
    topic_prompt = input("Enter a topic prompt: ")
    
    topic_title = generate_topic_title(topic_prompt)
    print(f"Generated topic title: {topic_title}")
    
    document_titles = generate_document_titles(topic_title)
    print("Generated document titles:")
    for title in document_titles:
        print(f"- {title}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{topic_title.replace(' ', '_')}"
    os.makedirs(folder_name, exist_ok=True)
    
    for title in document_titles:
        if not title:  # Skip empty titles
            logger.warning("Skipping empty title")
            continue
        try:
            content = generate_document_content(title)
            file_name = f"{title.replace(' ', '_')}.txt"
            file_path = os.path.join(folder_name, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Generated document: {file_path}")
        except Exception as e:
            logger.error(f"Error generating content for title '{title}': {e}")

if __name__ == "__main__":
    main()