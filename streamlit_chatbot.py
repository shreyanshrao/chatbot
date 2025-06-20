import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load model and tokenizer
MODEL_NAME = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)

# Sliding window conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Format chat history
def build_prompt(history, user_input):
    full_prompt = ""
    for pair in history[-3:]:  # Only last 3 exchanges
        full_prompt += f"User: {pair['user']}\nBot: {pair['bot']}\n"
    full_prompt += f"User: {user_input}\n"
    return full_prompt

# Get model response
def get_response(user_input):
    prompt = build_prompt(st.session_state.history, user_input)
    inputs = tokenizer([prompt], return_tensors="pt")
    result = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(result[0], skip_special_tokens=True)
    return response

# UI
st.title("🤖 BlenderBot Chatbot")

user_input = st.text_input("You:", key="input")

if st.button("Send") and user_input.strip():
    response = get_response(user_input)
    st.session_state.history.append({"user": user_input, "bot": response})
    st.markdown(f"**🤖 Bot:** {response}")

# Show chat history
if st.session_state.history:
    st.markdown("### Chat History (Last 3)")
    for pair in st.session_state.history[-3:]:
        st.write(f"🧑 You: {pair['user']}")
        st.write(f"🤖 Bot: {pair['bot']}")
