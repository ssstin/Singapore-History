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

# Page configuration
st.set_page_config(
    page_title="Singapore History Chatbot",
    page_icon="💬",
    layout="wide"
)

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
            collection_name = "singapore_history"  # Adjust this to your actual collection name
            collection = chroma_client.get_collection(collection_name)
            return collection
        except Exception as e:
            st.error(f"Error getting collection: {e}")
            return None
            
    except Exception as e:
        st.error(f"Error setting up vector store: {e}")
        return None

# Function to query your custom model
def query_huggingface_model(prompt, max_retries=2):
    # Get API key from Streamlit secrets
    api_token = st.secrets["HF_API_TOKEN"]
    
    # Your custom model URL
    API_URL = "https://api-inference.huggingface.co/models/ssstin/unsloth"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Format the prompt for your specific model
    prompt_with_context = (
        f"You are a helpful assistant specialized in Singapore history. "
        f"Keep your answers factual, informative and focused on Singapore's history.\n\n"
        f"Question: {prompt}\n\n"
        f"Answer:"
    )
    
    # Prepare the payload
    payload = {
        "inputs": prompt_with_context,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        },
        "options": {
            "use_cache": True,
            "wait_for_model": True
        }
    }
    
    # Add retry logic
    for attempt in range(max_retries):
        try:
            # Make the API request
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            # Log response for debugging
            print(f"Status code: {response.status_code}")
            print(f"Response preview: {response.text[:300]}")
            
            # Check if the request was successful
            if response.status_code == 200:
                try:
                    # Process the response
                    response_json = response.json()
                    
                    # Handle different response formats
                    if isinstance(response_json, list) and len(response_json) > 0:
                        # Format for text generation models
                        generated_text = response_json[0].get("generated_text", "")
                    elif isinstance(response_json, dict):
                        # Format for some models
                        generated_text = response_json.get("generated_text", "")
                    else:
                        # Fallback
                        generated_text = str(response_json)
                    
                    return generated_text.strip()
                        
                except Exception as e:
                    # Log the error and response for debugging
                    print(f"Error parsing response: {e}")
                    print(f"Response content: {response.text[:300]}...")
                    return f"I encountered an error processing the response. Please try again."
            
            # If model is loading, wait and retry
            elif response.status_code == 503:
                if attempt < max_retries - 1:
                    time.sleep(15)
                    continue
                else:
                    return "The model is currently initializing. Please try again shortly."
            else:
                if response.status_code == 403:
                    print(f"API Error: {response.status_code}")
                    print(f"Response details: {response.text}")
                    return "Access denied (HTTP 403). Your API token may not have permission to use this model."

                return f"Sorry, I encountered an error (Status code: {response.status_code}). Please try again later."
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            else:
                return "The request timed out. Please try again in a moment."
        except Exception as e:
            return f"An error occurred: {str(e)}. Please try again later."
    
    # If we've exhausted all retries
    return "Unable to get a response after multiple attempts. Please try again later."

# Function to perform RAG
def query_with_rag(prompt, collection, max_retries=2):
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
            
            # Query the model with the augmented prompt
            return query_huggingface_model(rag_prompt, max_retries)
        else:
            # Fallback to direct query if no results found
            return query_huggingface_model(prompt, max_retries)
    
    except Exception as e:
        st.error(f"RAG error: {e}")
        # Fallback to direct query if RAG fails
        return query_huggingface_model(prompt, max_retries)

# Setup sidebar
with st.sidebar:
    st.title("Singapore History Chatbot")
    
    # Model info
    st.subheader("About")
    st.write("This chatbot uses a custom Singapore History model hosted on Hugging Face to answer questions about Singapore's history.")
    
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
                        # Use RAG if vector store is available
                        if st.session_state.vector_store:
                            response = query_with_rag(user_input, st.session_state.vector_store)
                        else:
                            # Fallback to direct query
                            response = query_huggingface_model(user_input)
                        
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