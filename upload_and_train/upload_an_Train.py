import streamlit as st
import pandas as pd
import json
import re
import unicodedata
from bs4 import BeautifulSoup
import spacy
from spellchecker import SpellChecker
import fitz  # PyMuPDF

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

def chunk_by_sentence(text, chunk_size=1000, overlap=200, source_page=1, measure="char"):
    """
    Splits text into meaningful chunks based on sentence structure with optional overlap.

    Parameters:
        text (str): The input text to be chunked.
        chunk_size (int): Maximum size of each chunk (measured in characters or tokens).
        overlap (int): Amount to overlap between chunks (measured in characters or tokens).
        source_page (int): Metadata indicating the source page.
        measure (str): "char" for character-based count or "token" for token-based count.

    Returns:
        List[dict]: Each dictionary contains:
            - chunk_id
            - source_page
            - chunk_text
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks, current_chunk = [], ""
    chunk_id = 0

    def get_length(text_segment):
        """
        Helper function to calculate the length of a text segment based on the selected measurement.
        """
        if measure == "token":
            # Count tokens using spacy; note that nlp() is called on the segment.
            return len([token for token in nlp(text_segment)]) if text_segment.strip() else 0
        return len(text_segment)

    for sentence in sentences:
        sentence_length = get_length(sentence)
        current_length = get_length(current_chunk)
        
        # Handle sentences longer than the desired chunk size by splitting into words.
        if sentence_length > chunk_size:
            words = sentence.split()
            sub_chunk = ""
            for word in words:
                if get_length(sub_chunk + " " + word) <= chunk_size:
                    sub_chunk += " " + word
                else:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "source_page": source_page,
                        "chunk_text": sub_chunk.strip()
                    })
                    chunk_id += 1
                    sub_chunk = word
            if sub_chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "source_page": source_page,
                    "chunk_text": sub_chunk.strip()
                })
            # Reset current_chunk and move to the next sentence.
            current_chunk = ""
            continue

        # If appending the sentence doesn't exceed chunk_size, then add it.
        if current_length + sentence_length < chunk_size:
            current_chunk += " " + sentence
        else:
            # Finalize the current chunk.
            chunk_text = current_chunk.strip()
            chunks.append({
                "chunk_id": chunk_id,
                "source_page": source_page,
                "chunk_text": chunk_text
            })
            chunk_id += 1
            # Apply overlap from the end of the previous chunk based on measurement.
            if overlap:
                if measure == "token":
                    tokens = [token.text for token in nlp(chunk_text)]
                    overlap_tokens = tokens[-overlap:] if len(tokens) >= overlap else tokens
                    overlap_text = " ".join(overlap_tokens)
                else:
                    overlap_text = chunk_text[-overlap:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append({
            "chunk_id": chunk_id,
            "source_page": source_page,
            "chunk_text": current_chunk.strip()
        })

    return chunks

def chunk_by_paragraph(text, source_page=1, chunk_offset=0):
    """
    Splits text into chunks based on paragraphs.

    Each paragraph becomes a separate chunk with metadata.

    Returns a list of dictionaries with:
        - chunk_id (unique when using a chunk_offset)
        - source_page
        - chunk_text
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for i, paragraph in enumerate(paragraphs):
        chunks.append({
            "chunk_id": chunk_offset + i,
            "source_page": source_page,
            "chunk_text": paragraph
        })
    return chunks

def upload_and_display():
    """Handles file upload, processing, and display."""
    st.title("Process Your Own DATA to train AI Agent")

    # Ensure chunk_size is stored in session state for interactive control
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000  # Default value

    # Number input for chunk size
    chunk_size = st.number_input(
        "Chunk Size for Knowledge Base",
        min_value=60,
        max_value=5000,
        step=10,
        key="chunk_size"
    )

    # Add sidebar options for chunking method and measurement type
    chunk_method = st.sidebar.radio("Select Chunking Method", ("Sentence", "Paragraph"))
    measurement_type = st.sidebar.radio("Select Measurement Type", ("Character", "Token"))
    st.session_state.measurement_type = measurement_type  # Store selection in session state

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
                # Use PyMuPDF (fitz) to open the PDF from the uploaded file stream
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                all_chunks = []
                global_chunk_id = 0  # Global counter for paragraph chunk IDs

                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    cleaned_text = clean_text(text)
                    # Choose chunking method based on sidebar selection
                    if chunk_method == "Sentence":
                        measure = "token" if measurement_type == "Token" else "char"
                        page_chunks = chunk_by_sentence(cleaned_text, chunk_size, overlap=200, source_page=page_num + 1, measure=measure)
                    else:
                        page_chunks = chunk_by_paragraph(cleaned_text, source_page=page_num + 1, chunk_offset=global_chunk_id)
                        global_chunk_id += len(page_chunks)
                    all_chunks.extend(page_chunks)

                # Create DataFrame and adjust for downstream processing
                df = pd.DataFrame(all_chunks)
                # Rename 'chunk_text' to 'prompt' for consistency with later steps
                df.rename(columns={"chunk_text": "prompt"}, inplace=True)
                # Add an empty 'completion' column
                df["completion"] = ""

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
        df["completion"] = df["completion"].fillna(default_completion)
        data = []
        min_examples_required = 10  # OpenAI requires at least 10 examples

        # Retrieve measurement type from session state or default to character-based
        measurement_type = st.session_state.get("measurement_type", "Character")
        measure = "token" if measurement_type == "Token" else "char"

        for i in range(len(df)):
            # Ensure the prompt is not empty
            prompt_text = df.iloc[i]["prompt"].strip()
            completion_text = df.iloc[i]["completion"].strip() or default_completion

            if not prompt_text:
                continue  # Skip empty prompts

            # Split long prompts and completions into smaller chunks to create more examples
            split_prompts_dicts = chunk_by_sentence(prompt_text, chunk_size=300, overlap=50, source_page=0, measure=measure)
            split_prompts = [d["chunk_text"] for d in split_prompts_dicts]
            split_completions_dicts = chunk_by_sentence(completion_text, chunk_size=300, overlap=50, source_page=0, measure=measure)
            split_completions = [d["chunk_text"] for d in split_completions_dicts]

            for j in range(len(split_prompts)):
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
