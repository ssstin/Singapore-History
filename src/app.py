__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import time
from datetime import datetime
import requests
import json
import os
import chromadb
from huggingface_hub import hf_hub_download, login
import tempfile
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Page configuration
st.set_page_config(
    page_title="Singapore History Chatbot",
    page_icon="💬",
    layout="wide"
)

# Add this function to load the model
@st.cache_resource
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/phi-3.5-mini-instruct-bnb-4bit",
        device_map="cpu",
        load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained("unsloth/phi-3.5-mini-instruct-bnb-4bit")
    
    # Download your adapter
    adapter_path = hf_hub_download(
        repo_id="ssstin/unsloth",
        filename="adapter_config.json",
        repo_type="model"
    )
    adapter_dir = os.path.dirname(adapter_path)
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    return model, tokenizer

# Function to generate text with local model
def generate_with_local_model(prompt, max_new_tokens=512, temperature=0.7):
    # Load model if not already loaded
    if "model" not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.model, st.session_state.tokenizer = load_model()
    
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9
        )
    
    # Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load CSS from external file
def load_css(css_file):
    with open(css_file, "r") as f:
        css = f.read()
    return css

# Load CSS File
try:
    css = load_css("src/styles/main.css")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except Exception as e:
    st.write(f"CSS file not found. Default styling will be used. Error: {e}")

# Function to set up the vector store
@st.cache_resource
def setup_vector_store():
    # Get API token
    api_token = st.secrets["HF_API_TOKEN"]
    
    # Login to Hugging Face
    login(token=api_token)
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Download the vector store folder from Hugging Face
        vector_store_dir = os.path.join(temp_dir, "vector_store")
        os.makedirs(vector_store_dir, exist_ok=True)
        
        # Download the SQLite database file
        db_path = hf_hub_download(
            repo_id="ssstin/unsloth",
            filename="vector_store/chroma.sqlite3",
            repo_type="model",
            local_dir=vector_store_dir
        )
        
        # Download other necessary vector store files
        for filename in ["header.bin", "data_level0.bin", "length.bin", "link_lists.bin", "index_metadata.pickle"]:
            try:
                file_path = hf_hub_download(
                    repo_id="ssstin/unsloth",
                    filename=f"vector_store/e3b5bd58-d969-4f41-a6b7-cf2da0a9283c/{filename}",
                    repo_type="model",
                    local_dir=os.path.join(vector_store_dir, "e3b5bd58-d969-4f41-a6b7-cf2da0a9283c")
                )
            except Exception as e:
                print(f"Could not download {filename}: {e}")
        
        # Initialize Chroma client
        chroma_client = chromadb.PersistentClient(path=vector_store_dir)
        
        # Get the collection
        try:
            # Try to get the existing collection
            collection_name = "e3b5bd58-d969-4f41-a6b7-b8501f795315"  
            collection = chroma_client.get_collection(collection_name)
            return collection
        except Exception as e:
            st.error(f"Error getting collection: {e}")
            return None
            
    except Exception as e:
        st.error(f"Error setting up vector store: {e}")
        return None

# Modified RAG function to use local model
def query_with_rag(prompt, collection):
    try:
        # Query the vector store
        results = collection.query(
            query_texts=[prompt],
            n_results=3
        )
        
        # Extract context from results
        if results and "documents" in results and len(results["documents"]) > 0:
            contexts = results["documents"][0]
            context_text = "\n".join(contexts)
            
            # Create the augmented prompt with context
            rag_prompt = (
                f"Context information:\n{context_text}\n\n"
                f"Based on this context, answer the following question about Singapore history:\n"
                f"Question: {prompt}\n\n"
                f"Answer:"
            )
            
            # Use local model
            return generate_with_local_model(rag_prompt)
        else:
            # Fallback to direct query if no results found
            return generate_with_local_model(prompt)
    
    except Exception as e:
        st.error(f"RAG error: {e}")
        # Fallback to direct query if RAG fails
        return generate_with_local_model(prompt)

# Setup sidebar
with st.sidebar:
    st.title("Singapore History Chatbot")
    
    # Model info
    st.subheader("About")
    st.write("This chatbot uses a custom Singapore History model to answer questions about Singapore's history.")
    
    st.divider()
    
    # Add a button to start a new chat
    if st.button("Start New Chat"):
        # Reset messages to initial state
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Singapore History guide. What would you like to know about Singapore's past?", "timestamp": datetime.now().strftime("%H:%M")}
        ]
        st.rerun()
    
    st.divider()
    
    # Tips for users
    st.subheader("Tips")
    st.write("Try asking about:")
    st.write("• The founding of Singapore")
    st.write("• Major historical events")
    st.write("• Cultural heritage")
    st.write("• Singapore's path to independence")

# Initialize vector store on app start
if "vector_store" not in st.session_state:
    with st.spinner("Setting up knowledge base..."):
        st.session_state.vector_store = setup_vector_store()

# Initialize model on startup (optional, can be lazy-loaded later)
if "model" not in st.session_state and "tokenizer" not in st.session_state:
    try:
        with st.spinner("Loading model (this may take a moment)..."):
            st.session_state.model, st.session_state.tokenizer = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.session_state.model_error = True

# Set up the main content area
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # Title bar
    st.markdown("<div class='title-container'><h1>Singapore History Chatbot</h1></div>", unsafe_allow_html=True)
    
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Singapore History guide. What would you like to know about Singapore's past?", "timestamp": datetime.now().strftime("%H:%M")}
        ]

    # Display chat messages
    chat_container = st.container()
    
    # Input area (fixed at bottom)
    input_container = st.container()
    
    # Display chat messages from history
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-container user-container">
                    <div class="message-header">You</div>
                    <div class="timestamp">{message["timestamp"]}</div>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-container bot-container">
                    <div class="message-header">Singapore History Chatbot</div>
                    <div class="timestamp">{message["timestamp"]}</div>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Add some space at the bottom to prevent overlap with input
        st.markdown("<div style='height: 100px'></div>", unsafe_allow_html=True)
    
    # User input area
    with input_container:
        st.markdown("<div class='input-container'>", unsafe_allow_html=True)
        
        # Create a form for the input to prevent automatic reloading
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.text_input("Ask me about Singapore's history...", key="user_input")
            
            with col2:
                submit_button = st.form_submit_button("Send")
            
            if submit_button and user_input:
                # Add user message to chat history
                st.session_state.messages.append(
                    {"role": "user", "content": user_input, "timestamp": datetime.now().strftime("%H:%M")}
                )
                
                # Display typing indicator while waiting for response
                with st.spinner("Thinking..."):
                    try:
                        # Check if model loaded successfully
                        if hasattr(st.session_state, 'model_error') and st.session_state.model_error:
                            response = "I'm sorry, there was an error loading the model. Please try again later."
                        else:
                            # Use RAG if vector store is available
                            if st.session_state.vector_store:
                                response = query_with_rag(user_input, st.session_state.vector_store)
                            else:
                                # Fallback to direct query without RAG
                                response = generate_with_local_model(user_input)
                        
                        # Add AI response to chat history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response, "timestamp": datetime.now().strftime("%H:%M")}
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        # Add error message to chat history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"Sorry, I encountered an error. Please try again later.", 
                             "timestamp": datetime.now().strftime("%H:%M")}
                        )
                
                # Rerun the app to display the updated chat
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

# Footer with information
st.markdown("""
<div style="text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 12px; color: #888;">
    This is a demonstration project using a custom Singapore History model.
</div>
""", unsafe_allow_html=True)