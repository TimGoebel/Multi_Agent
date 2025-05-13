import streamlit as st
from upload_and_train.upload_an_Train import upload_and_display, preprocess_data, save_training_file
from fine_tune.fine_tune import fine_tune_model
from chat.chat_with_model import chat_with_model
from vlm_image.text_vlm import vlm_gbt
from vlm_video.vlm_video_action import vlm_gbt
from utils.utils import load_restricted_words
from evaluate.evaluate_model import evaluate_fine_tuned_model  # Import the new evaluation function

def main():
    st.sidebar.title("Navigation")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

    # Load restricted words and available models
    restricted_list, model_list = load_restricted_words()
    base_model = st.sidebar.selectbox("Select Base Model for Fine-Tuning", options=model_list)

    choice = st.sidebar.radio("Go to", ["Upload & Process Data", "Train Model", "Evaluate Model", "Chat with AI","VLM_image","VLM_video"])

    if not api_key:
        st.sidebar.warning("Please enter your OpenAI API Key to proceed.")
        return

    if choice == "Upload & Process Data":
        df = upload_and_display()
        if df is not None:
            data = preprocess_data(df)
            if data is not None:
                save_training_file(data)
    elif choice == "Train Model":
        if st.button("Start Training"):
            fine_tune_model("training_data.jsonl", api_key, base_model)
    elif choice == "Evaluate Model":
        st.title("Evaluate Fine-Tuned Model")
        # model_id = st.text_input("Enter Fine-Tuned Model ID:", help="Find this in your OpenAI fine-tuning results.")
        model_id = st.selectbox("Enter Fine-Tuned Model ID:", options=model_list)
        if st.button("Evaluate Model"):
            if model_id:
                evaluate_fine_tuned_model(api_key, base_model)
            else:
                st.warning("Please enter a valid Fine-Tuned Model ID.")
    elif choice == "Chat with AI":
        chat_with_model(api_key, restricted_list, model_list)

    elif choice == "VLM_image" and base_model == "gpt-4-turbo":
        vlm_gbt(api_key,base_model)

    elif choice == "VLM_video" and base_model == "gpt-4-turbo":
        vlm_gbt(api_key,base_model)


if __name__ == "__main__":
    main()
