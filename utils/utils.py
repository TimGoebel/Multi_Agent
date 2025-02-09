import json

def load_restricted_words(file_path="restricted_words.json"):
    """Loads restricted words and available fine-tuned models from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data.get("restricted_words", []), data.get("model_list", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return [], []
