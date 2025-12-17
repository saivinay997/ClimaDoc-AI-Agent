import streamlit as st


st.title("ClimaDoc - Your WeatherRAGAi agent!!")

def sidebar_and_documentChooser():
    with st.sidebar:
        st.caption(
            "ClimaDoc AI is an intelligent AI agent that combines real-time weather intelligence with document-based question answering"
        )
        st.write("")
        
        
def clear_chat_history():
    """clear chat history and memory."""
    # 1. re-initialize messages
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "How can I assist you today?",
        }
    ]
    # 2. Clear memory (history)
    try:
        st.session_state.memory.clear()
    except:
        pass


def chatbot():
    sidebar_and_documentChooser()
    st.divider()
    col1, col2 = st.columns([7,3])
    with col1:
        st.subheader("Chat with your data")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history)

