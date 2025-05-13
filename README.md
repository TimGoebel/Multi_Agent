# Multi-Agent AI Assistant 🚀

This project is a **Streamlit-based AI Assistant** that allows users to **train, fine-tune, and chat with AI models** using OpenAI's API.

## 📌 Features
- **Upload and Process Data** (PDF, CSV, JSON)
- **Preprocess Data** for fine-tuning
- **Evaluate Model** for model evaluation tool
- **Fine-tune AI Models** with OpenAI's API
- **Multi-Agent AI Chat** with customizable prompts
- **Restricted Words Filtering** to control AI responses
- **Vision-Language Model (VLM) for AI-powered image analysis** 🖼️

## 🏗️ Project Structure

```
your_project/
│── main.py                    # Main entry point for Streamlit app
│── upload_and_train/
│   ├── upload_an_Train.py      # Handles file upload and preprocessing
│── fine_tune/
│   ├── fine_tune.py            # Fine-tuning OpenAI models
│── evaluate/
│   ├── evaluate_model.py       # Evaulate OpenAI models to the new trained model
│── chat/
│   ├── chat_with_model.py      # AI chatbot interactions
│── vlm/
│   ├── text_vlm.py             # AI Vision AI agent
│── utils/
│   ├── __init__.py             # Marks `utils/` as a package
│   ├── utils.py                # Contains helper functions
│── restricted_words.json       # Stores restricted words & model list (optional)
│── requirements.txt            # Python dependencies
│── README.md                   # Project documentation
```

## 📦 Installation
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/TimGoebel/Multi_Agent.git
cd Multi_Agent
```

### **2️⃣ Set Up a Virtual Environment (Recommended)**
```sh
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 🚀 Running the App
Run the Streamlit app with:
```sh
streamlit run main.py
```

or use **run.bat** (for Windows users):

```
@echo off
python -m streamlit run main.py
pause  # Remove this if you don't want the CMD screen to stay open for troubleshooting
```

---

## **🛠️ Data Processing Pipeline**
This project **prepares and processes textual data** from **PDF, CSV, and JSON files** for fine-tuning **GPT models**. It ensures high-quality, structured training data by applying **advanced text cleaning, chunking, and formatting**. The pipeline guarantees a minimum of **10 well-structured training examples**, which is required for OpenAI fine-tuning.

### **1️⃣ Upload & Extract Data**
- **Supported Formats:** `PDF`, `CSV`, `JSON`
- **File Upload:** Users can upload training data via **Streamlit’s file uploader**.
- **Extraction Process:**
  - `CSV` → Loaded into a DataFrame.
  - `JSON` → Parsed as structured text.
  - `PDF` → Text is extracted **page by page** using `PyPDF2`.

---

### **2️⃣ Advanced Text Cleaning**
To improve model quality, raw text is cleaned using the following techniques:
- **Unicode Normalization:** Fixes encoding inconsistencies (`unicodedata.normalize("NFKC", text)`).
- **HTML Tag Removal:** Removes any embedded HTML content (`BeautifulSoup`).
- **Hyphenation Fixes:** Merges words split across lines (`cross-\nword → crossword`).
- **Extra Whitespace Removal:** Normalizes spacing (`re.sub(r"\s+", " ", text)`).
- **Smart Quotes & Symbol Conversion:** Converts to standard ASCII quotes and dashes (`“” → "`, `— → -`).
- **Spell Checking & Correction:** Fixes common spelling errors (`SpellChecker`).

---

### **3️⃣ Text Chunking & Structuring**

To prevent **mid-sentence breaks**, the extracted text is **split into meaningful units** using `spaCy`:

* **NLP Sentence-Based Chunking:** Instead of breaking text arbitrarily at `chunk_size`, it ensures **semantic coherence**.
* **Adaptive Chunking:** If a chunk exceeds `1000` characters, it is **split into smaller, logical parts**.

The UI supports flexible chunking and measurement settings:

* **Chunking Method:** Choose between Sentence-level or Paragraph-level segmentation.
* **Measurement Type:** Select how chunk length is measured — by Character count or Token count.


---

### **4️⃣ Formatting for GPT-3 Fine-Tuning**
Each data sample is formatted as a **conversational exchange**:
```json
{
  "messages": [
    {"role": "system", "content": "You are an AI trained on specialized data."},
    {"role": "user", "content": "How does Mobile Lock protect authentication?"},
    {"role": "assistant", "content": "Mobile Lock prevents unauthorized access by restricting authentication when threats are detected."}
  ]
}
```
- **Ensures well-structured conversations**.
- **Prevents empty "user" messages**.
- **Pairs each prompt with a valid assistant response**.

---

### **5️⃣ Ensuring Minimum of 10 Examples**
- If the dataset contains **fewer than 10 training pairs**, additional examples are **generated automatically**.
- Large text blocks are **split into multiple prompts**, ensuring a **diverse dataset**.
- **Default responses are used** where assistant completions are missing.

---

### **6️⃣ Exporting to JSONL for Fine-Tuning**
The final dataset is saved in **JSONL format**, ready for OpenAI fine-tuning:
```sh
training_data.jsonl
```
Each entry is saved **on a new line** to comply with OpenAI’s fine-tuning requirements.

---

## **📌 How to Use**
### **1️⃣ Upload & Train**
- Upload a **PDF, CSV, or JSON** file.
- The data is chunked and preprocessed for fine-tuning.

### **2️⃣ Fine-Tune Model**
- Train a new model using OpenAI’s fine-tuning API.
- Monitors training progress until completion.

### **3️⃣ Save & Train**
- Click **"Save Training Data"** to export `training_data.jsonl`.
- Use OpenAI’s fine-tuning API to train the model.

---

## **🚀 Fine-Tuning Instructions**
### **1️⃣ Prepare the Training Data**
Ensure the dataset is ready in `JSONL` format:
```sh
openai tools fine_tunes.prepare_data -f training_data.jsonl
```

### **2️⃣ Train the Model**
Fine-tune using OpenAI’s API:
```sh
openai api fine_tunes.create -t training_data.jsonl -m gpt-3.5-turbo
```

### **3️⃣ Deploy and Test**
Use the fine-tuned model in your AI assistant.

---

## 🏗️ Future Enhancements
- Add **local model fine-tuning** support.
- add **embedings**
- Expand AI assistant **multi-modal capabilities**.
- Integrate **vector search** for improved retrieval.

## 📜 License
This project is licensed under the **MIT License**.

---

### 🔗 Connect with Me
- **GitHub:** [Timothy Goebel](https://github.com/TimGoebel)
- **LinkedIn:** [Timothy Goebel](http://www.linkedin.com/in/timothygoebel)

