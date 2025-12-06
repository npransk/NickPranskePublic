import streamlit as st
import sys
import traceback

# Page configuration
st.set_page_config(
    page_title="The Shadow and the Stag Book Club Chatbot",
    layout="centered"
)

st.title("The Shadow and the Stag Book Club Chatbot")

# Show Python and package versions for debugging
with st.expander("Debug Info"):
    st.write(f"Python version: {sys.version}")
    try:
        import torch
        st.write(f"PyTorch version: {torch.__version__}")
    except:
        st.write("PyTorch: Not loaded yet")

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    import base64
    import os
    
    st.success("✓ Basic imports successful")
    
    @st.cache_resource
    def load_epub_from_secrets():
        """Load epub file from Streamlit secrets (base64 encoded)"""
        try:
            epub_base64 = st.secrets["epub"]["shadow_and_stag"]
            epub_bytes = base64.b64decode(epub_base64)
            
            temp_path = "/tmp/temp_book.epub"
            with open(temp_path, "wb") as f:
                f.write(epub_bytes)
            
            book = epub.read_epub(temp_path)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return book
        except KeyError:
            st.error("Could not find 'shadow_and_stag' in secrets.")
            return None
        except Exception as e:
            st.error(f"Error loading book: {e}")
            st.code(traceback.format_exc())
            return None

    @st.cache_resource
    def extract_text_from_epub(_book):
        """Extract all text content from epub"""
        if _book is None:
            return []
        
        chapters = []
        for item in _book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                if text:
                    chapters.append(text)
        
        return chapters

    @st.cache_resource
    def split_into_chunks(chapters, chunk_size=500):
        """Split text into smaller chunks"""
        chunks = []
        for chapter in chapters:
            words = chapter.split()
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if chunk:
                    chunks.append(chunk)
        return chunks

    @st.cache_resource
    def load_models():
        """Load embedding model and LLM"""
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        import warnings
        warnings.filterwarnings('ignore')
        
        st.info("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        st.success("✓ Embedding model loaded")
        
        st.info("Loading language model (this may take a few minutes)...")
        # Use a smaller, more reliable model
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        
        # Set pad token
        tokenizer.pad_token = tokenizer.eos_token
        
        st.success("✓ Language model loaded")
        
        return embedding_model, model, tokenizer

    @st.cache_data
    def create_embeddings(_embedding_model, chunks):
        """Create embeddings for all chunks"""
        embeddings = _embedding_model.encode(chunks, show_progress_bar=False)
        return embeddings

    def retrieve_relevant_chunks(query, _embedding_model, _embeddings, chunks, top_k=3):
        """Retrieve most relevant chunks for a query"""
        import numpy as np
        query_embedding = _embedding_model.encode([query])
        similarities = np.dot(_embeddings, query_embedding.T).squeeze()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [chunks[i] for i in top_indices]

    def generate_response(_model, _tokenizer, query, context_chunks):
        """Generate response using LLM with retrieved context"""
        context = " ".join(context_chunks[:3])  # Use top 3 chunks, join without newlines
        
        # Simple prompt that encourages direct answers
        prompt = f"""Using only this information: {context[:800]}

Question: {query}
Direct answer:"""

        inputs = _tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=400, 
            truncation=True,
            padding=True
        )
        
        outputs = _model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.85,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id
        )
        
        response = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer after "Direct answer:"
        if "Direct answer:" in response:
            answer = response.split("Direct answer:")[-1].strip()
        elif "answer:" in response.lower():
            answer = response.split("answer:")[-1].strip()
        else:
            answer = response.strip()
            
        # Clean up only excessive newlines
        answer = answer.replace("\n\n", " ").replace("\n", " ").strip()
            
        return answer

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Load everything
    st.info("Step 1/5: Loading book from secrets...")
    book = load_epub_from_secrets()
    
    if book is None:
        st.stop()
    
    st.success("✓ Book loaded")
    
    st.info("Step 2/5: Extracting content...")
    chapters = extract_text_from_epub(book)
    st.success(f"✓ Extracted {len(chapters)} chapters")
    
    st.info("Step 3/5: Creating chunks...")
    chunks = split_into_chunks(chapters)
    st.success(f"✓ Created {len(chunks)} chunks")
    
    st.info("Step 4/5: Loading AI models (2-3 minutes first time)...")
    embedding_model, model, tokenizer = load_models()
    
    st.info("Step 5/5: Creating index...")
    embeddings = create_embeddings(embedding_model, chunks)
    st.success(f"🎉 Ready! {len(chunks)} chunks indexed")
    
    st.markdown("---")
    st.markdown("Ask questions about the book below:")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if query := st.chat_input("Ask a question about the book"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                try:
                    relevant_chunks = retrieve_relevant_chunks(
                        query, embedding_model, embeddings, chunks
                    )
                    response = generate_response(model, tokenizer, query, relevant_chunks)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.code(traceback.format_exc())

except Exception as e:
    st.error(f"Critical error: {e}")
    st.code(traceback.format_exc())