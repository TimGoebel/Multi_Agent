import streamlit as st
import pandas as pd
import json
import re
import unicodedata
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import spacy
from spellchecker import SpellChecker

# Load NLP model for better sentence chunking
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

def clean_text(text):
    """Performs advanced text cleaning on extracted PDF content."""
    if not text:  # Handle NoneType or empty strings
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove special symbols except common punctuation
    text = re.sub(r"[^\w\s,.!?;:()'-]", "", text)

    # Fix hyphenated words from line breaks
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # Merge hyphenated words
    text = text.replace("-\n", "")  # Remove any remaining hyphen line breaks

    # Remove extra spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Convert smart quotes & dashes to ASCII
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")

    # Ensure spell checker doesn't return None
    words = text.split()
    text = " ".join([spell.correction(word) if spell.correction(word) is not None else word for word in words])

    return text

def chunk_by_sentence(text, chunk_size=1000):
    """Splits text into meaningful chunks based on sentence structure."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def upload_and_display():
    """Handles file upload, processing, and display."""
    st.title("Train Your Own AI Agent")

    # Ensure chunk_size is stored in session state for interactive control
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000  # Default value

    chunk_size = st.number_input(
        "Chunk Size for Knowledge Base",
        min_value=60,
        max_value=5000,
        value=st.session_state.chunk_size,
        step=10,
        key="chunk_size"
    )

    uploaded_file = st.file_uploader("Upload your training data (PDF, CSV, or JSON)", type=["pdf", "csv", "json"])

    if uploaded_file is not None:
        processing_message = st.empty()
        processing_message.info("Processing file...")

        file_extension = uploaded_file.name.split(".")[-1].lower()
        df = None  # Initialize DataFrame

        try:
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)

            elif file_extension == "json":
                df = pd.read_json(uploaded_file)

            elif file_extension == "pdf":
                pdf_reader = PdfReader(uploaded_file)

                # Extract and clean text safely (handle NoneType)
                text = "\n".join([clean_text(page.extract_text() or "") for page in pdf_reader.pages])

                # Chunk the cleaned text using NLP-based sentence splitting
                text_chunks = chunk_by_sentence(text, chunk_size)

                # Create DataFrame for GPT-3 fine-tuning
                df = pd.DataFrame({"prompt": text_chunks, "completion": [""] * len(text_chunks)})

            else:
                st.error("Unsupported file format.")
                return None

        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None

        processing_message.success("File processing complete!")
        st.write("Preview of uploaded data:")
        st.dataframe(df)
        return df

    return None

def preprocess_data(df):
    """Convert DataFrame into a format suitable for fine-tuning."""
    st.write("Preprocessing Data...")

    default_completion = st.sidebar.text_input("Default Completion Text", value="This is the default completion text.")

    if "prompt" in df.columns:
        df["completion"].fillna(default_completion, inplace=True)

        data = []
        min_examples_required = 10  # OpenAI requires at least 10 examples

        for i in range(len(df)):
            # Ensure the prompt is not empty
            prompt_text = df.iloc[i]["prompt"].strip()
            completion_text = df.iloc[i]["completion"].strip() or default_completion

            if not prompt_text:
                continue  # Skip empty prompts

            # Split long prompts into smaller chunks to create more examples
            split_prompts = chunk_by_sentence(prompt_text, chunk_size=300)  # Adjusted chunk size
            split_completions = chunk_by_sentence(completion_text, chunk_size=300)

            for j in range(len(split_prompts)):
                # Create a message pair
                entry = {
                    "messages": [
                        {"role": "system", "content": "You are an AI trained on specialized data."},
                        {"role": "user", "content": split_prompts[j]},
                        {"role": "assistant", "content": split_completions[j] if j < len(split_completions) else default_completion}
                    ]
                }
                data.append(entry)

        # Ensure at least 10 examples
        while len(data) < min_examples_required:
            data.append({
                "messages": [
                    {"role": "system", "content": "You are an AI trained on specialized data."},
                    {"role": "user", "content": f"Generated example {len(data) + 1}"},
                    {"role": "assistant", "content": default_completion}
                ]
            })

        st.success(f"Preprocessing complete! Total examples: {len(data)}")
        return data

    else:
        st.error("Data must contain 'prompt' and 'completion' columns.")
        return None


def save_training_file(data):
    """Save the preprocessed data into a JSONL file for fine-tuning."""
    filename = "training_data.jsonl"
    
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)  # Ensures UTF-8 encoding
            f.write("\n")  # Write each entry on a new line

    return filename
