# Email Assistant

A local email assistant that allows you to query your Outlook emails using natural language.

## Features

- Extract emails from Outlook
- Save emails and attachments locally
- Create a searchable database of your emails
- Query your emails using natural language
- Chat interface for easy interaction

## Setup

1. Install dependencies:

   pip install -r requirements.txt



2. Extract emails from Outlook:

   python email_reader.py


Follow the prompts to specify the date range.

3. Download the Llama model:

   python llama_model.py


Follow the instructions to download the model file.

4. Create the vector database:

   python vector_store.py



5. Start the chat interface:

   streamlit run app.py



## Usage

Once the app is running, you can ask questions about your emails such as:

- "What did John email me about last month?"
- "Find emails about the marketing project"
- "When was the last time I received an email from HR?"
- "Summarize my conversations with Sarah"

## Project Structure

- `email_reader.py`: Extracts emails from Outlook
- `llama_model.py`: Sets up the Llama language model
- `vector_store.py`: Creates and manages the vector database
- `app.py`: Streamlit chat interface
- `models/`: Directory for storing the Llama model
- `data/`: Directory for storing emails, attachments, and the vector database
