# LLM Knowledge Distillation and RAG Testing Tool

## Overview

This AI-powered document generator is designed to create a corpus of documents on a given topic, leveraging large language models (LLMs) to distill knowledge into a format suitable for training or fine-tuning smaller models. It serves as a platform for testing and comparing various Retrieval-Augmented Generation (RAG) solutions and configurations, as well as for knowledge distillation experiments.

## Features

- Generate a topic title based on a user-provided prompt
- Create multiple document titles related to the main topic
- Generate comprehensive content for each document title in YAML format
- Support for both Anthropic API and Open API Standard Services (e.g., OpenWebUI)
- Configurable number of documents to generate
- Progress tracking and logging
- Easily adaptable for different LLM APIs

## Requirements

- Python 3.7+
- Required Python packages (install using `pip install -r requirements.txt`):
  - requests
  - tqdm

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/christian-taillon/llm-knowledge-distillation-tool.git
   cd llm-knowledge-distillation-tool
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API key:
   - For Anthropic API: Set the `ANTHROPIC_API_KEY` environment variable
   - For Open API Standard Service: Set the `OPENWEBUI_KEY` environment variable (if required)

## Usage

Run the script:

```
python distillery.py
```

Follow the prompts to:
1. Choose between Anthropic API or Open API Standard Service
2. Enter a topic prompt
3. Specify the number of documents to generate

The script will create a new folder with the generated documents in YAML format.

## Configuration

You can modify the following variables in the script:

- `MODEL`: The AI model to use (default is "claude-3-sonnet-20240229" for Anthropic or "llama3:405b" for Open API Standard)
- `API_ENDPOINT`: The API endpoint URL

## Research Background

### Knowledge Distillation for LLMs

Knowledge distillation is a technique used to transfer knowledge from a large, complex model (the "teacher") to a smaller, more efficient model (the "student"). This process is crucial in the field of LLMs for several reasons:

1. **Efficiency**: Smaller models require less computational resources and can run on more modest hardware.
2. **Speed**: Distilled models often have faster inference times, making them more suitable for real-time applications.
3. **Privacy**: Smaller models can sometimes be run on-device, reducing the need to send data to external servers.

This tool aids in the first two steps of this process by generating a corpus of documents that capture the knowledge of a large LLM on a specific topic.

### Retrieval-Augmented Generation (RAG)

RAG is a technique that combines the strengths of retrieval-based and generation-based approaches in natural language processing. It involves:

1. Retrieving relevant information from a knowledge base.
2. Using this information to augment the context provided to a language model.
3. Generating responses based on both the original query and the retrieved information.

This approach helps to ground the model's outputs in factual information and can improve the accuracy and relevance of generated content.

## Using This Tool for RAG Testing

This tool can be used to create custom knowledge bases for RAG systems. By generating a corpus of documents on specific topics, you can:

1. Test different retrieval algorithms to see which ones most effectively find relevant information.
2. Experiment with various ways of incorporating retrieved information into prompts.
3. Compare the performance of different LLMs when given additional context through RAG.
4. Evaluate the impact of knowledge base size and diversity on RAG performance.

## Customization

- Modify the API request functions to use different LLM APIs.
- Adjust the prompts in `generate_topic_title`, `generate_document_titles`, and `generate_document_content` to tailor the output to your specific needs.
- Extend the script to include automatic evaluation metrics for generated content.

## Logging

The script logs its activities to a file in the `logs` directory. Each run creates a new log file with a timestamp.

## Contributing

Contributions to improve the tool or extend its capabilities are welcome. Please submit a pull request or open an issue to discuss proposed changes.
