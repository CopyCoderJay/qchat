import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Get Hugging Face token from environment
HF_TOKEN = os.getenv("HF_TOKEN", None)

if not HF_TOKEN:
    st.error("⚠️ HF_TOKEN not found in .env file! Please add your HuggingFace token.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="English Grammar & Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "General Chat"

if "client" not in st.session_state:
    with st.spinner("Connecting to HuggingFace API..."):
        try:
            st.session_state.client = InferenceClient(
                api_key=HF_TOKEN,
            )
            st.success("✅ Connected to HuggingFace API!")
        except Exception as e:
            st.error(f"Failed to connect: {str(e)}")
            st.stop()

# Mode configurations
MODE_PROMPTS = {
    "General Chat": {
        "system_prompt": "You are a helpful and friendly English learning assistant. Help the user with their questions in a clear and engaging way.",
        "icon": "💬"
    },
    "Grammar Checker": {
        "system_prompt": "You are an expert English grammar teacher. Analyze the user's text for grammar errors, explain the mistakes, and provide corrected versions with clear explanations.",
        "icon": "✏️"
    },
    "Paragraph Writer": {
        "system_prompt": "You are a creative writing assistant. Help users write well-structured, coherent paragraphs on various topics. Provide multiple style options (formal, casual, academic) when appropriate.",
        "icon": "📝"
    },
    "Python Helper": {
        "system_prompt": "You are a Python programming tutor. Help users understand Python concepts, debug code, write better code, and follow best practices. Provide clear explanations with examples.",
        "icon": "🐍"
    }
}

def get_response(user_input, mode):
    """Get response from the HuggingFace API"""
    try:
        system_prompt = MODE_PROMPTS[mode]["system_prompt"]
        
        # Try Qwen first, fallback to other models if not available
        models_to_try = [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Llama-3.2-3B-Instruct",
            "google/gemma-2-2b-it"
        ]
        
        for model in models_to_try:
            try:
                completion = st.session_state.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                )
                
                response = completion.choices[0].message.content
                return response
            except Exception as model_error:
                # Try next model if this one fails
                if model != models_to_try[-1]:  # Don't log error for intermediate failures
                    continue
                raise model_error
                
    except Exception as e:
        error_msg = str(e)
        if "model_pending_deploy" in error_msg or "not ready for inference" in error_msg:
            return "⚠️ The model is warming up. Please wait a moment and try again. This happens when the model hasn't been used recently on the HuggingFace servers.\n\n**Tip:** Try clicking your message again in a few seconds."
        elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return "❌ Authentication error. Please check your HF_TOKEN in the .env file."
        else:
            return f"Error generating response: {str(e)}"

# Sidebar for mode selection
with st.sidebar:
    st.title("🎓 Learning Modes")
    
    for mode_name, mode_config in MODE_PROMPTS.items():
        if st.button(f"{mode_config['icon']} {mode_name}", key=f"mode_{mode_name}"):
            st.session_state.current_mode = mode_name
    
    st.markdown("---")
    st.markdown(f"**Current Mode:** {MODE_PROMPTS[st.session_state.current_mode]['icon']} {st.session_state.current_mode}")
    
    st.markdown("---")
    st.markdown("### Model Information")
    st.info("""
    **Models:** Qwen2.5-1.5B-Instruct (with fallbacks)
    **Mode:** Cloud API
    
    Running via HuggingFace Inference API:
    - Auto-tries multiple models
    - No local model download
    - May need "warming" time
    """)
    
    st.warning("⚠️ If you see 'model warming' errors, wait 10-15 seconds and try again. The free tier needs to wake up the model.")
    
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main content area
st.markdown('<div class="main-header">📚 English Grammar & Learning Assistant</div>', unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about English grammar, writing, or Python coding..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt, st.session_state.current_mode)
        st.markdown(response)
    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #808080;'>
        Powered by Qwen2.5-1.5B-Instruct via HuggingFace API | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

