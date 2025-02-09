import streamlit as st
import openai
import os
import time
import json
import requests
from openai import OpenAI

def fine_tune_model(training_file, api_key, base_model, model_list):
    st.title("Train Model")
    client = OpenAI(api_key=api_key)

    abs_training_file = os.path.abspath(training_file)
    
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(abs_training_file, "rb") as f:
        files = {"file": (os.path.basename(abs_training_file), f)}
        data = {"purpose": "fine-tune"}
        response = requests.post("https://api.openai.com/v1/files", headers=headers, data=data, files=files)
    
    if response.status_code != 200:
        st.error("Error uploading file for fine-tuning:")
        st.error(response.text)
        return None

    file_id = response.json().get("id")
    if not file_id:
        st.error("File upload did not return an ID.")
        return None

    try:
        job_data = client.fine_tuning.jobs.create(training_file=file_id, model=base_model)
    except Exception as e:
        st.error("Error starting fine-tuning:")
        st.error(e)
        return None

    job_id = job_data.id
    st.write(f"Fine-tuning started successfully. Job ID: {job_id}")

    while True:
        try:
            status_data = client.fine_tuning.jobs.retrieve(job_id)
        except Exception as e:
            st.error("Error retrieving fine-tuning job status:")
            st.error(e)
            return None
        
        job_status = status_data.status
        st.write(f"Current status: {job_status}")
        
        if job_status in ["succeeded", "failed", "cancelled"]:
            break
        time.sleep(30)

    if job_status == "succeeded":
        fine_tuned_model = status_data.fine_tuned_model or "Unknown"
        st.success(f"Fine-tuning completed successfully! Model: {fine_tuned_model}")
    else:
        st.error(f"Fine-tuning failed. Status: {job_status}")

    return status_data
