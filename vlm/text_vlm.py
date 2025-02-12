import streamlit as st
import openai
from PIL import Image
import base64
from io import BytesIO
import os

# 🔄 Function to encode image as base64 for GPT-4 Turbo
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# 🤖 Function to analyze an uploaded image with GPT-4 Turbo
def analyze_image_with_gpt4_turbo(image_path, prompt, api_key, base_model):
    client = openai.OpenAI(api_key=api_key)
    try:
        with open(image_path, "rb") as file:
            image_bytes = file.read()
            base64_image = encode_image(image_bytes)

        response = client.chat.completions.create(
            model=base_model,
            messages=[
                {"role": "system", "content": "You are an AI assistant analyzing images."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=300
        )

        return response.choices[0].message.content  

    except openai.OpenAIError as e:
        return f"❌ OpenAI API Error: {str(e)}"

# ✅ FIXED FUNCTION WITHOUT CLIP
def vlm_gbt(api_key, base_model):
    """Vision-Language Model UI with GPT-4 Turbo"""

    # 🎨 Streamlit UI
    st.title("🖼️ Vision-Language Model with GPT-4 Turbo")
    st.write("Upload an image and enter a prompt to analyze it.")

    uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])
    prompt = st.text_input("💬 Enter a prompt", "Describe this image")

    if uploaded_file is not None:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("🔎 Analyzing..."):
                gpt4_response = analyze_image_with_gpt4_turbo(temp_path, prompt, api_key, base_model)

            # 🔥 Display Results
            st.subheader("📖 GPT-4 Turbo Analysis")
            st.write(gpt4_response)
