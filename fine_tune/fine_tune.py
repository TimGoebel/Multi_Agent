import streamlit as st
import openai
import os
import time
import requests
import json  # ✅ Import json module
from openai import OpenAI

def fine_tune_model(training_file, api_key, base_model, model_list=None):
    st.title("Train Model with Fine-Tuning")
    client = OpenAI(api_key=api_key)

    abs_training_file = os.path.abspath(training_file)
    
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(abs_training_file, "rb") as f:
        files = {"file": (os.path.basename(abs_training_file), f)}
        data = {"purpose": "fine-tune"}
        
        try:
            response = requests.post("https://api.openai.com/v1/files", headers=headers, data=data, files=files)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Error uploading file for fine-tuning: {str(e)}")
            return None

    file_id = response.json().get("id")
    if not file_id:
        st.error("File upload did not return an ID.")
        return None

    try:
        job_data = client.fine_tuning.jobs.create(training_file=file_id, model=base_model)
    except Exception as e:
        st.error(f"Error starting fine-tuning: {str(e)}")
        return None

    job_id = job_data.id
    st.success(f"Fine-tuning started successfully. Job ID: {job_id}")

    progress_bar = st.progress(0)

    while True:
        try:
            status_data = client.fine_tuning.jobs.retrieve(job_id)
            job_status = status_data.status
            st.write(f"Current status: {job_status}")

            if job_status == "succeeded":
                fine_tuned_model = status_data.fine_tuned_model or "Unknown"
                st.success(f"Fine-tuning completed successfully! Model: {fine_tuned_model}")

                # Update a local JSON file (e.g., "restricted_words.json") with the new model name.
                if fine_tuned_model != "Unknown":
                    try:
                        with open("restricted_words.json", "r") as file:
                            json_data = json.load(file)
                    except (FileNotFoundError, json.JSONDecodeError):
                        json_data = {"restricted_words": [], "model_list": []}
                    
                    if fine_tuned_model not in json_data.get("model_list", []):
                        json_data["model_list"].append(fine_tuned_model)
                    
                    with open("restricted_words.json", "w") as file:
                        json.dump(json_data, file, indent=4)

                break
            elif job_status in ["failed", "cancelled"]:
                st.error(f"Fine-tuning failed. Status: {job_status}")
                break

            progress_bar.progress(min(1.0, time.time() % 100 / 100))
            time.sleep(30)  # Polling interval

        except Exception as e:
            st.error(f"Error retrieving fine-tuning job status: {str(e)}")
            return None

    return status_data
