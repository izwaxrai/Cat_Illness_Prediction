import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from datetime import datetime
import json
import pandas as pd

# --- Page Setup ---
st.set_page_config(page_title="Cat Health Chatbot", page_icon="🐾")

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .chat-container {
        background-color: #f0f2f6;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        animation: fadeIn 0.5s ease-in-out;
    }
    .user-icon { color: #e0c3ab; margin-right: 8px; }
    .bot-icon { color: #e0c3ab; margin-right: 8px; }
    .message-bubble {
        padding: 8px 12px;
        border-radius: 8px;
        max-width: 80%;
    }
    .user-bubble { background-color: #a7c6d4; color: #3a71a6; }
    .bot-bubble { background-color: #bdd9bf; color: #4f6e50; }
    .timestamp { color: #757575; font-size: 0.8em; margin-top: 5px; }
    .main-title { color: #333; text-align: center; margin-bottom: 10px; }
    .subtitle { color: #777; text-align: center; margin-bottom: 20px; font-size: 0.9em; }
    .feedback-btn {
        cursor: pointer;
        font-size: 20px;
        margin-left: 10px;
        color: gray;
        transition: color 0.3s ease;
        border: none;
        background: none;
    }
    .feedback-btn:hover {
        color: #4f6e50;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    div.stButton > button[kind] {
        background-color: #b4cced !important;
        color: #405778 !important;
        border: 1px solid #778599 !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
        width: auto !important;
    }
    div.stButton > button[kind]:hover {
        background-color: #778599 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load Q&A data from CSV
@st.cache_data
def load_qa_data():
    df = pd.read_csv("cat_illness_augmented_data.csv")
    df.dropna(subset=[df.columns[0], df.columns[1]], inplace=True)
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))  # Q -> A

qa_data = load_qa_data()


# --- Load Model & Tokenizer ---
@st.cache_resource
def load_models():
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        lora_model = PeftModel.from_pretrained(base_model, "cat_health_lora_model")
        lora_model.eval()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        lora_model = None

    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"❌ Error loading summarizer: {e}")
        summarizer = None

    return tokenizer, lora_model, summarizer

tokenizer, model, summarizer = load_models()


# --- Generate Answer ---
def generate_answer(prompt: str) -> str:
    # Check if user's question matches one in the CSV
    if prompt in qa_data:
        return qa_data[prompt]

    if model is None:
        return "The model is not available right now."

    # Otherwise use the model
    input_text = f"You are a helpful veterinary assistant. Answer clearly.\nQuestion: {prompt}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024)

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=60,
            top_p=0.95,
            max_length=300,
            min_length=40,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("Answer:", "").strip()

    # Keyword-based filtering
    irrelevant_keywords = ["insurance", "pregnancy", "birth control"]
    if any(keyword in answer.lower() for keyword in irrelevant_keywords):
        answer = "Sorry, that doesn't seem relevant. Here's how to keep your cat healthy: ensure regular vet checkups, a balanced diet, vaccinations, and a clean environment."

    return answer + "\n\n📚 For more, check the Advanced VetCare Veterinary Centre or consult your local vet."


# --- Summarize Answer ---
def summarize_answer(text: str) -> str:
    if summarizer is None:
        return "Summary unavailable."
    try:
        summary = summarizer(
            text,
            max_length=150,
            min_length=40,
            do_sample=True,
            clean_up_tokenization_spaces=True
        )
        return summary[0]["summary_text"]
    except Exception:
        return "Summary unavailable."


# --- Initialize Chat ---
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize feedback storage for each bot message by index in history
if "feedback" not in st.session_state:
    # dict: key = index of bot answer in history, value = {"up": int, "down": int}
    st.session_state.feedback = {}

st.markdown("<h1 class='main-title'>🐱 Cat Health Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask about symptoms, care, or wellness tips for your cat!</p>", unsafe_allow_html=True)

example_questions = [
    "How can I keep my cat from getting sick?",
    "What are some common signs that my cat might be unwell?",
    "My cat isn't eating. What should I do?",
    "How can I tell if my cat has a fever?",
    "How often should I take my cat to the vet?",
    "Why is my cat sneezing a lot?",
    "Why is my cat shedding so much?",
    "What are the symptoms of kidney disease in cats?",
    "What are the signs of a serious health problem in cats?",
    "What are signs of a cat emergency?",
    "What are common symptoms of illness in cats?",
    "How can I prevent common cat illnesses?"
]

with st.sidebar:
    st.header("👋 Try asking me...")
    for i, q in enumerate(example_questions):
        if st.button(q, key=f"example_{i}"):
            st.session_state.selected_example = q
            st.rerun()

if "selected_example" in st.session_state:
    user_input = st.session_state.selected_example
    del st.session_state.selected_example
    process_input = True
else:
    user_input = st.text_input("Ask me about your cat:", key="user_input")
    process_input = st.button("Send")

if process_input:
    if model is not None and user_input.strip():
        with st.spinner("Thinking..."):
            try:
                answer = generate_answer(user_input)
                summary = summarize_answer(answer)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.session_state.history.append(("user", user_input, timestamp))
                st.session_state.history.append(("bot", answer, timestamp))
                st.session_state.history.append(("summary", summary, timestamp))

                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Something went wrong: {e}")
    elif not user_input.strip():
        st.warning("Please type your question!")
    elif model is None:
        st.error("Hmm, I seem to be having a little trouble connecting right now.")

# --- Chat Filtering / Search ---
search_query = st.text_input("🔍 Search chat history", key="chat_search").strip().lower()

st.markdown("### 💬 Chat", unsafe_allow_html=True)

for idx, (role, message, time) in enumerate(st.session_state.history):
    # Apply filter if search query exists
    if search_query and search_query not in message.lower():
        continue

    if role == "summary":
        icon = "🧠"
        bubble_class = "bot-bubble"
        role_label = "Summary"
        # Summaries usually no feedback needed
        show_feedback = False
    elif role == "user":
        icon = "👤"
        bubble_class = "user-bubble"
        role_label = "You"
        show_feedback = False
    else:
        icon = "🐱"
        bubble_class = "bot-bubble"
        role_label = "Bot"
        show_feedback = True

    # Highlight emergency messages
    if "see a vet" in message.lower() or "emergency" in message.lower():
        message = f"<span style='color:black'><strong>⚠️ {message}</strong></span>"

    st.markdown(f"""
    <div class='chat-container'>
        <span class='{ "user-icon" if role == "user" else "bot-icon" }'>{icon}</span>
        <div class='message-bubble {bubble_class}'>
            <strong>{role_label}:</strong><br>{message}
            <div class='timestamp'>{time}</div>
    """, unsafe_allow_html=True)

    # --- Feedback buttons for bot answers ---
    if show_feedback:
        # Initialize feedback counts if missing
        if idx not in st.session_state.feedback:
            st.session_state.feedback[idx] = {"up": 0, "down": 0}

        col1, col2, col3 = st.columns([1,1,10])
        with col1:
            if st.button("👍", key=f"up_{idx}"):
                st.session_state.feedback[idx]["up"] += 1
                st.rerun()
        with col2:
            if st.button("👎", key=f"down_{idx}"):
                st.session_state.feedback[idx]["down"] += 1
                st.rerun()
        with col3:
            up = st.session_state.feedback[idx]["up"]
            down = st.session_state.feedback[idx]["down"]
            st.markdown(f"👍 {up}   👎 {down}")

    st.markdown("</div></div>", unsafe_allow_html=True)

# --- Download Chat ---
def download_chat():
    return json.dumps([{"role": r, "message": m, "timestamp": t} for r, m, t in st.session_state.history], indent=4).encode("utf-8")

if st.session_state.history:
    st.download_button("Download Chat", download_chat(), file_name="chat_history.json", mime="application/json")
