import torch
import streamlit as st
import json
from transformers import pipeline

# Load translation model
text_translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", torch_dtype=torch.bfloat16)

# Load the JSON data from the file
with open('language.json', 'r') as file:
    language_data = json.load(file)

# Extract all language names for the dropdown
all_languages = [entry["Language"] for entry in language_data]

# Function to get FLORES code from language
def get_FLORES_code_from_language(language):
    for entry in language_data:
        if entry['Language'].lower() == language.lower():
            return entry['FLORES-200 code']
    return None

# Function to chunk the input text if it exceeds token limits
def chunk_text(text, chunk_size=512):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > chunk_size:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to translate text
def translate_text(text, destination_language):
    # Get the language code
    dest_code = get_FLORES_code_from_language(destination_language)
    if not dest_code:
        return "Selected language is not available in the translation model."
    
    # Split long text into smaller chunks
    chunks = chunk_text(text)
    translations = []
    
    for chunk in chunks:
        try:
            # Translate each chunk and collect results
            translation = text_translator(chunk, src_lang="eng_Latn", tgt_lang=dest_code)
            translations.append(translation[0]["translation_text"])
        except Exception as e:
            translations.append(f"Error during translation: {str(e)}")
    
    return " ".join(translations)

# Streamlit UI
st.title("@GenAILearniverse Project 4: Multi-language Translator")
st.write("This application will translate any English text into multiple languages.")

# Text input
text_to_translate = st.text_area("Input text to translate", height=150)

# Dropdown for language selection
destination_language = st.selectbox("Select Destination Language", all_languages)

if st.button("Translate"):
    if text_to_translate:
        with st.spinner("Translating..."):
            translated_text = translate_text(text_to_translate, destination_language)
            st.subheader("Translated Text:")
            st.write(translated_text)
    else:
        st.warning("Please enter some text to translate.")
