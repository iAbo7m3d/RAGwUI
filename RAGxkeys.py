import streamlit as st
import traceback
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List

# --- Configuration via user input ---
st.sidebar.header("🔑 أدخل مفاتيح API")
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
PINECONE_API_KEY = st.sidebar.text_input("Pinecone API Key", type="password")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.sidebar.warning("يرجى إدخال مفاتيح OpenAI و Pinecone للاستمرار.")
    st.stop()

INDEX_NAME = "labor-law-v2"

# Initialize embedding model, Pinecone client, and OpenAI client
embedder = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
client = OpenAI(api_key=OPENAI_API_KEY)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "مرحبًا! كيف يمكنني مساعدتك في فهم قوانين العمل بالسعودية؟"}
    ]


def retrieve_context_simple(query: str, top_k: int = 10) -> List[str]:
    # 1) Encode the query
    q_vec = embedder.encode(query, normalize_embeddings=True).tolist()
    # 2) Retrieve top_k matches
    resp = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
    matches = resp.get("matches", [])
    # 3) Extract content
    return [m["metadata"].get("content", "") for m in matches]


def generate_response_simple(query: str) -> str:
    
    chunks = retrieve_context_simple(query)
    if not chunks:
        return "عذراً، لا توجد معلومات متاحة لإجابة هذا السؤال."
    context = "\n\n".join(chunks)
    messages = [
        {"role": "system", "content": (
            "أنت مساعد ذكي تساعد المستخدمين على فهم أنظمة العمل في المملكة العربية السعودية. "
            " أجب بدقة وباللغة العربية بناءً على المعلومات المقدمة."
            "اذكر نص المادة ورقمها التي استندت عليها في بناء اجابتك "
            "اجتنب ذكر المواد والنصوص التي ليست لها علاقة مباشرة بالسؤال"
        )},
        {"role": "user", "content": f"المعلومات المتاحة:\n{context}\n\nالسؤال: {query}"}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception:
        traceback.print_exc()
        return "عذراً، حدث خطأ أثناء المعالجة. يرجى المحاولة مرة أخرى."


def main():
    st.title("🤖 المستشار الآلي")

    with st.sidebar:
        st.header("📚 دليل المستخدم")
        st.markdown("""
        ### طريقة الاستخدام
        1. **ابدأ المحادثة**: اكتب سؤالك في صندوق الدردشة في الأسفل.
        2. **استمر في الحوار**: الأداة تتذكر المحادثة، ويمكنك طرح أسئلة متابعة.
        3. **عرض المحادثات السابقة**: قم بالتمرير للأعلى لرؤية تاريخ الدردشة.
        """)
        if st.button("🔄 إعادة تعيين المحادثة"):
            st.session_state.messages = [
                {"role": "assistant", "content": "مرحبًا! كيف يمكنني مساعدتك في فهم قوانين العمل بالسعودية؟"}
            ]
            st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_input := st.chat_input("ما هو سؤالك حول قوانين العمل؟"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                reply = generate_response_simple(user_input)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
