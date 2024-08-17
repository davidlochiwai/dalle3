import streamlit as st
import os
import requests
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI
import json
from dotenv import load_dotenv
import utils

# Login Page
utils.setup_page("Image Generation")

# Load environment variables for OpenAI API configurations
load_dotenv()

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_version="2024-02-01",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
)

def generate_image(prompt, size, quality, style):
    result = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality=quality,
        style=style,
        n=1
    )
    json_response = json.loads(result.model_dump_json())
    image_url = json_response["data"][0]["url"]
    image = Image.open(BytesIO(requests.get(image_url).content))
    return image

def refine_prompt(description, usage):
    # Use GPT-4 model to refine the prompt
    response = client.chat.completions.create(
        model="gpt-4o-128K",
        messages=[{"role": "system", "content": "You are a helpful assistant."},{"role":"user","content": f"Generate a detailed prompt for an image based on your best understanding on the following user's description and usage:\nDescription: {description}\nUsage: {usage}\nSpecify photorealistic style if the user did not specify any. Only return with the prompt texts."}],
        max_tokens=500
    )
    refined_prompt = response.choices[0].message.content
    return refined_prompt

def main():
    st.title("Image Generation with DALL-E-3")

    # Sidebar options
    size = st.sidebar.radio("Select Image Size", ["1024x1024", "1792x1024", "1024x1792"], index=0)
    quality = "standard"
    style = "vivid"

    # quality = st.sidebar.radio("Select Image Quality", ["standard", "hd"], index=0)
    # style = st.sidebar.radio("Select Image Style", ["vivid", "natural"], index=0)

    if 'description' not in st.session_state:
        st.session_state['description'] = ''
    if 'usage' not in st.session_state:
        st.session_state['usage'] = ''
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = ''
    if 'image' not in st.session_state:
        st.session_state['image'] = None

    if st.session_state['image'] is None:
        description = st.text_input("Describe the image you want to generate:", key='description_input')
        usage = st.text_input("What will you use the image for?", key='usage_input')
        if st.button("Refine Prompt"):
            with st.spinner("Refining prompt..."):
                st.session_state['description'] = description
                st.session_state['usage'] = usage
                st.session_state['prompt'] = refine_prompt(description, usage)
                st.experimental_rerun()
        
        if st.session_state['prompt']:
            prompt = st.text_area("Refined Prompt (you can edit this):", st.session_state['prompt'], key='prompt_input')
            if st.button("Generate Image"):
                with st.spinner("Generating image..."):
                    st.session_state['prompt'] = prompt
                    st.session_state['image'] = generate_image(prompt, size, quality, style)
                    st.experimental_rerun()
    else:
        st.image(st.session_state['image'], caption="Generated Image", use_column_width=True)
        prompt = st.text_area("Refined Prompt (you can edit this):", st.session_state['prompt'], key='prompt_input')
        if st.button("Generate Image"):
            with st.spinner("Generating image..."):
                st.session_state['prompt'] = prompt
                st.session_state['image'] = generate_image(prompt, size, quality, style)
                st.experimental_rerun()
        if st.button("Reset"):
            st.session_state['description'] = ''
            st.session_state['usage'] = ''
            st.session_state['prompt'] = ''
            st.session_state['image'] = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()