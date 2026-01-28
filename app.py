import os
import gradio as gr
from groq import Groq

# -----------------------------
# Load API Key (HF only)
# -----------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")

client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.1-8b-instant"

# -----------------------------
# Chat Logic
# -----------------------------
def chat_with_groq(message, history):
    messages = []

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )

    return response.choices[0].message.content


def respond(message, history):
    history = history or []
    
    history.append({"role": "user", "content": message})
    
    bot_response = chat_with_groq(message, [
        (h["content"], history[i+1]["content"])
        for i, h in enumerate(history[:-1])
        if h["role"] == "user"
    ])
    
    history.append({"role": "assistant", "content": bot_response})
    
    return "", history


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Groq Chatbot")
    chatbot = gr.Chatbot(type="messages")

    msg = gr.Textbox(placeholder="Ask me anything...")
    clear = gr.Button("Clear Chat")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()
