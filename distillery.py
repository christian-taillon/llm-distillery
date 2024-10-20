import os
import requests
import json
from datetime import datetime
import logging
from tqdm import tqdm

# Set up logging to write to a file
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def choose_api():
    while True:
        choice = input("Choose API (1 for Anthropic, 2 for Open API Standard): ").strip()
        if choice == '1':
            return 'anthropic'
        elif choice == '2':
            return 'open_api'
        else:
            print("Invalid choice. Please enter 1 or 2.")

def get_model_name(api_type):
    if api_type == 'anthropic':
        return "claude-3-sonnet-20240229"
    else:
        model = input("Enter model name (default is llama3:405b): ").strip()
        return model if model else "llama3:405b"

def setup_api(api_type, model):
    if api_type == 'anthropic':
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        api_endpoint = 'https://api.anthropic.com/v1/messages'
        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }
    else:
        api_key = os.environ.get('OPENWEBUI_KEY')
        api_endpoint = 'https://openwebui-host.local/api/chat/completions'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    return api_endpoint, headers, model

def api_request(prompt, api_endpoint, headers, model):
    logger.info(f"Making API request to model: {model}")
    if 'anthropic-version' in headers:
        data = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 1000
        }
    else:
        data = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ]
        }
    
    try:
        response = requests.post(api_endpoint, headers=headers, json=data)
        response.raise_for_status()
        logger.debug(f"API Response: {response.text}")
        logger.info("API request completed")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        logger.error(f"Response content: {response.text if response else 'No response'}")
        raise

def generate_topic_title(topic_prompt, api_endpoint, headers, model):
    prompt = f"Generate a concise topic title (2-4 words) based on the following description: {topic_prompt}"
    response = api_request(prompt, api_endpoint, headers, model)
    if 'content' in response:
        return response['content'][0]['text'].strip()
    else:
        return response['choices'][0]['message']['content'].strip()

def generate_document_titles(topic_title, count, api_endpoint, headers, model):
    prompt = f"""
    Based on the topic "{topic_title}", generate a list of {count} document titles that would be valuable for an LLM to reference in a Retrieval-Augmented Generation (RAG) system. These titles should cover various aspects and subtopics related to the main topic. 
    Provide only the titles, one per line, without numbering or additional explanation.
    Avoid using colons (:) in the titles.
    """
    response = api_request(prompt, api_endpoint, headers, model)
    if 'content' in response:
        titles = response['content'][0]['text'].strip().split('\n')
    else:
        titles = response['choices'][0]['message']['content'].strip().split('\n')
    return [title.strip().replace(':', '-') for title in titles if title.strip()]

def generate_document_content(title, api_endpoint, headers, model):
    prompt = f"""
    Create a comprehensive document on the topic: "{title}"
    This document will be used as a reference for an LLM in a Retrieval-Augmented Generation (RAG) system. 
    Include relevant information, key concepts, examples, and any other pertinent details that would be valuable for an AI to understand and utilize this topic.
    The document should be well-structured, informative, and approximately 500-1000 words in length.
    Output the content in YAML format.
    """
    response = api_request(prompt, api_endpoint, headers, model)
    if 'content' in response:
        return response['content'][0]['text'].strip()
    else:
        return response['choices'][0]['message']['content'].strip()

def main():
    api_type = choose_api()
    model = get_model_name(api_type)
    api_endpoint, headers, model = setup_api(api_type, model)

    topic_prompt = input("Enter a topic prompt: ")
    doc_count = int(input("Enter the number of documents to generate (default is 5): ") or "5")
    
    topic_title = generate_topic_title(topic_prompt, api_endpoint, headers, model)
    print(f"Generated topic title: {topic_title}")
    
    document_titles = generate_document_titles(topic_title, doc_count, api_endpoint, headers, model)
    print("Generated document titles:")
    for title in document_titles:
        print(f"- {title}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{topic_title.replace(' ', '_').replace(':', '-')}"
    os.makedirs(folder_name, exist_ok=True)
    
    print("\nGenerating documents:")
    pbar = tqdm(total=len(document_titles), ncols=70, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    
    for title in document_titles:
        if not title:
            logger.warning("Skipping empty title")
            continue
        try:
            pbar.set_description(f"Generating: {title[:30]}...")
            content = generate_document_content(title, api_endpoint, headers, model)
            file_name = f"{title.replace(' ', '_').replace(':', '-')}.yaml"
            file_path = os.path.join(folder_name, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated document: {file_path}")
            pbar.update(1)
        except Exception as e:
            logger.error(f"Error generating content for title '{title}': {e}")
    
    pbar.close()
    print(f"\nGeneration complete. Log file: {log_file}")

if __name__ == "__main__":
    main()