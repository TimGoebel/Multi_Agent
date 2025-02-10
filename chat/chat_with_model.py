import streamlit as st
import re
from openai import OpenAI

def get_openai_response(prompt, api_key, restricted_list, chat_model):
    """
    Fetches a response from the OpenAI API using the ChatCompletion endpoint
    and filters out any restricted words.
    """
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=chat_model,  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    content = response.choices[0].message.content.strip()

    for word in restricted_list:
        content = content.replace(word, "[REDACTED]")
    return content

def chat_with_model(api_key, restricted_list, model_list):
    """Multi-Agent AI Assistant"""
    st.title("Multi-Agent AI Assistant 🤖")
    chat_model = st.selectbox("Chat model selection:", options=model_list)
    topic = st.text_area("Enter your topic:")

    if "viral_hooks" not in st.session_state:
        st.session_state.viral_hooks = []
    if "selected_hook" not in st.session_state:
        st.session_state.selected_hook = None
    if "content_generated" not in st.session_state:
        st.session_state.content_generated = False
    if "article" not in st.session_state:
        st.session_state.article = ""
    if "social_media_posts" not in st.session_state:
        st.session_state.social_media_posts = ""
    if "short_story" not in st.session_state:
        st.session_state.short_story = ""

    viral_hook_prompt_input = st.text_input("Enter a custom prompt for the viral hook:", value="Generate 10 viral hooks for: ")
    article_prompt_input = st.text_input("Enter a custom prompt for the blog post:", value="Write a short blog post about: ")
    social_media_prompt_input = st.text_input("Enter a custom prompt for social media posts:", value="Write 3 engaging social media posts about: ")
    story_prompt_input = st.text_input("Enter a custom prompt for the short story:", value="Create a short story about: ")

    tone_setting = (
        "The tone should show expertise, seriousness, and empathy. It should sound like it was written by a human. "
        "Keep each sentence to a maximum of 10-12 words. Add a lot of whitespace."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        generate_button = st.button("Generate Viral Hooks")
    with col2:
        clear_button = st.button("Clear All")

    if clear_button:
        st.session_state.clear()
        st.session_state.topic = ""
        st.session_state.viral_hooks = []
        st.session_state.selected_hook = None
        st.session_state.content_generated = False
        st.session_state.article = ""
        st.session_state.social_media_posts = ""
        st.session_state.short_story = ""

    if generate_button:
        if not api_key or not topic:
            st.error("Please enter both your API key and topic.")
        else:
            with st.spinner("Generating viral hooks..."):
                viral_hook_prompt = f"{viral_hook_prompt_input} {topic}. {tone_setting}"
                viral_hooks_raw = get_openai_response(viral_hook_prompt, api_key, restricted_list, chat_model)
                st.session_state.viral_hooks = re.findall(r'\d+\.\s*(.*)', viral_hooks_raw)
                st.session_state.viral_hooks = [hook.strip() for hook in st.session_state.viral_hooks if hook.strip()]
                if not st.session_state.viral_hooks:
                    st.warning("No viral hooks were generated. Please try again with a different topic.")

    if st.session_state.viral_hooks:
        st.subheader("Select the Best Viral Hook")
        st.session_state.selected_hook = st.radio("Choose one:", st.session_state.viral_hooks[:5])
        if st.button("Generate Content Based on Selected Hook"):
            if st.session_state.selected_hook:
                with st.spinner("Generating content..."):
                    article_prompt = f"{article_prompt_input} {st.session_state.selected_hook}. {tone_setting}"
                    social_media_prompt = f"{social_media_prompt_input} {st.session_state.selected_hook}. {tone_setting}"
                    story_prompt = f"{story_prompt_input} {st.session_state.selected_hook}. {tone_setting}"
                    st.session_state.article = get_openai_response(article_prompt, api_key, restricted_list, chat_model)
                    st.session_state.social_media_posts = get_openai_response(social_media_prompt, api_key, restricted_list, chat_model)
                    st.session_state.short_story = get_openai_response(story_prompt, api_key, restricted_list, chat_model)
                    st.session_state.content_generated = True

    if st.session_state.content_generated:
        st.subheader("Blog Post")
        st.write(st.session_state.article)
        st.subheader("Social Media Posts")
        st.write(st.session_state.social_media_posts)
        st.subheader("Short Story")
        st.write(st.session_state.short_story)
