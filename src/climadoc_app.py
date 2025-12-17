import streamlit as st
import os
import tempfile
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from workflow import run_climadoc_workflow
from rag_pipelines import document_ingestion_pipeline

# Page configuration
st.set_page_config(
    page_title="ClimaDoc AI Agent",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gemini-2.0-flash"

if "api_key" not in st.session_state:
    # Try to get API key from environment variable as default
    st.session_state.api_key = os.getenv("GOOGLE_API_KEY", "")

if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False


def initialize_llm():
    """Initialize the LLM with the API key and model from session state."""
    if st.session_state.api_key and st.session_state.llm_model:
        try:
            llm = ChatGoogleGenerativeAI(
                model=st.session_state.llm_model,
                google_api_key=st.session_state.api_key,
                temperature=0.7
            )
            st.session_state.llm_initialized = True
            return llm
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            st.session_state.llm_initialized = False
            return None
    return None


def clear_chat_history():
    """Clear chat history and conversation history."""
    st.session_state.messages = []
    st.session_state.conversation_history = []
    st.success("Chat history cleared!")


def sidebar_configuration():
    """Configure sidebar with LLM selection and API key input."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.divider()
        
        # LLM Model Selection
        st.subheader("LLM Settings")
        llm_options = [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro"
        ]
        
        selected_model = st.selectbox(
            "Select LLM Model",
            options=llm_options,
            index=llm_options.index(st.session_state.llm_model) if st.session_state.llm_model in llm_options else 0,
            help="Choose the Google Gemini model to use"
        )
        
        if selected_model != st.session_state.llm_model:
            st.session_state.llm_model = selected_model
            st.session_state.llm_initialized = False
        
        # API Key Input
        api_key = st.text_input(
            "Google API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your Google API key for Gemini models",
            placeholder="Enter your API key here"
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.llm_initialized = False
        
        # Initialize LLM button
        if st.button("Initialize LLM", type="primary", use_container_width=True):
            llm = initialize_llm()
            if llm:
                st.success("LLM initialized successfully!")
            else:
                st.error("Failed to initialize LLM. Please check your API key.")
        
        # Show initialization status
        if st.session_state.llm_initialized:
            st.success("✅ LLM Ready")
        else:
            st.warning("⚠️ LLM not initialized")
        
        st.divider()
        
        # Document Upload Section
        st.subheader("📄 Document Upload")
        st.caption("Upload PDF files to add them to the vector database")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF file to ingest into the vector database"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary", use_container_width=True):
                with st.spinner("Processing document..."):
                    try:
                        # Create temporary directory for the uploaded file
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)
                            file_path = temp_path / uploaded_file.name
                            
                            # Save uploaded file
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Process document
                            document_ingestion_pipeline(base_dir=str(temp_path))
                            st.success(f"✅ Document '{uploaded_file.name}' processed and added to vector database!")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
        
        st.divider()
        
        # Clear Chat History Button
        st.subheader("🗑️ Chat Management")
        if st.button("Clear Chat History", use_container_width=True, type="secondary"):
            clear_chat_history()
        
        st.divider()
        
        # Information Section
        st.subheader("ℹ️ About")
        st.caption(
            "ClimaDoc AI is an intelligent AI agent that combines real-time weather intelligence "
            "with document-based question answering. Upload documents and ask questions!"
        )


def display_chat_messages():
    """Display chat messages in the chat interface."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main():
    """Main application function."""
    # Title and header
    st.title("🌤️ ClimaDoc AI Agent")
    st.caption("Your intelligent weather and document assistant powered by AI")
    
    # Sidebar configuration
    sidebar_configuration()
    
    # Main chat interface
    st.divider()
    
    # Display chat messages
    display_chat_messages()
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about weather or your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to conversation history
        st.session_state.conversation_history.append(HumanMessage(content=prompt))
        
        # Get LLM instance (always try to initialize to handle API key changes)
        llm = initialize_llm()
        if not llm:
            error_msg = "⚠️ Please configure and initialize the LLM in the sidebar before asking questions."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.stop()
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Run the workflow
                    result = run_climadoc_workflow(
                        query=prompt,
                        conversation_history=st.session_state.conversation_history[:-1],  # Exclude current message
                        llm_instance=llm
                    )
                    
                    # Extract the final answer from the result
                    if result and "messages" in result and len(result["messages"]) > 0:
                        # Get the last message (should be the final answer)
                        final_message = result["messages"][-1]
                        
                        if hasattr(final_message, 'content'):
                            response_text = final_message.content
                        else:
                            response_text = str(final_message)
                        
                        # Display the response
                        st.markdown(response_text)
                        
                        # Add assistant message to chat
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        
                        # Add assistant message to conversation history
                        st.session_state.conversation_history.append(AIMessage(content=response_text))
                    else:
                        error_msg = "I apologize, but I couldn't generate a response. Please try again."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        
                except Exception as e:
                    error_msg = f"I encountered an error: {str(e)}. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.conversation_history.append(AIMessage(content=error_msg))


if __name__ == "__main__":
    main()
