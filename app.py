from openai import OpenAI
oai = OpenAI()  # otomatis baca OPENAI_API_KEY dari env/secrets

def call_openai(messages, model="gpt-4o-mini", temperature=0.7, top_p=1.0, max_tokens=512):
    resp = oai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content

import google.generativeai as genai
genai.configure()

def call_gemini(messages, model="gemini-1.5-flash", temperature=0.7, top_p=1.0, top_k=40, max_tokens=512):
    m = genai.GenerativeModel(model)
    history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})
    user_text = messages[-1]["content"]
    chat = m.start_chat(history=history)
    cfg = genai.types.GenerationConfig(
        temperature=temperature, top_p=top_p, top_k=top_k, max_output_tokens=max_tokens
    )
    return chat.send_message(user_text, generation_config=cfg).text

import streamlit as st
st.set_page_config(page_title="Mini Assignment C1", page_icon="💬")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Tanyakan apa saja 😊"}]

st.title("LLM Chat – Mini Assignment C1")

# tampilkan riwayat
for m in st.session_state.messages:
    with st.chat_message("assistant" if m["role"]=="assistant" else "user"):
        st.markdown(m["content"])

# input chat
user_msg = st.chat_input("Tulis pesanmu…")
if user_msg:
    st.session_state.messages.append({"role":"user","content":user_msg})

with st.sidebar:
    st.subheader("Model Settings")
    provider = st.selectbox("Provider", ["OpenAI","Gemini"])
    if provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-4o-mini","gpt-4.1-mini"], index=0)
    else:
        model_name = st.selectbox("Model", ["gemini-1.5-flash","gemini-1.5-pro"], index=0)

with st.sidebar:
    st.markdown("---")
    st.caption("Advanced settings")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    top_p       = st.slider("Top-P",       0.0, 1.0, 1.0, 0.05)
    top_k       = st.slider("Top-K (Gemini)", 0, 100, 40, 1)
    max_tokens  = st.slider("Max tokens (output)", 0, 4096, 512, 64)

if user_msg:
    with st.chat_message("assistant"):
        with st.spinner("Memikirkan jawaban…"):
            msgs = [{"role": x["role"], "content": x["content"]} for x in st.session_state.messages]
            if provider=="OpenAI":
                reply = call_openai(msgs, model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            else:
                reply = call_gemini(msgs, model=model_name, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens)
            st.markdown(reply)
    st.session_state.messages.append({"role":"assistant","content":reply})

def summarize_history(provider, model_name, temperature, top_p, top_k, max_tokens):
    convo = "\n".join(
        ["User: "+m["content"] if m["role"]=="user" else "Assistant: "+m["content"]
         for m in st.session_state.messages]
    )
    prompt = ("Ringkas percakapan berikut menjadi 5–7 poin bullet berbahasa Indonesia, "
              "fokus pada fakta/keputusan.\n\n"+convo)

    if provider=="OpenAI":
        return call_openai([{"role":"user","content":prompt}], model=model_name,
                           temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    else:
        return call_gemini([{"role":"user","content":prompt}], model=model_name,
                           temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens)

with st.sidebar:
    if st.button("📝 Summarize chat"):
        if st.session_state.messages:
            summary = summarize_history(provider, model_name, temperature, top_p, top_k, max_tokens)
            st.session_state.messages.append({"role":"assistant","content":summary})
            st.success("Ringkasan ditambahkan ke chat ✅")
        else:
            st.info("Belum ada percakapan untuk diringkas.")