import streamlit as st
from upload_and_train.upload_an_Train import upload_and_display, preprocess_data, save_training_file
from fine_tune.fine_tune import fine_tune_model
from chat.chat_with_model import chat_with_model
from utils.utils import load_restricted_words

def main():
    st.sidebar.title("Navigation")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

    # Load restricted words and available models
    restricted_list, model_list = load_restricted_words()
    base_model = st.sidebar.selectbox("Select Base Model for Fine-Tuning", options=model_list)
    
    choice = st.sidebar.radio("Go to", ["Upload & Train", "Train Data", "Chat with AI"])

    if not api_key:
        st.sidebar.warning("Please enter your OpenAI API Key to proceed.")
        return

    if choice == "Upload & Train":
        df = upload_and_display()
        if df is not None:
            data = preprocess_data(df)
            if data is not None:
                save_training_file(data)
    elif choice == "Train Data":
        if st.button("Start Training"):
            fine_tune_model("training_data.jsonl", api_key, base_model, model_list)
    elif choice == "Chat with AI":
        chat_with_model(api_key, restricted_list, base_model)

if __name__ == "__main__":
    main()
