import streamlit as st
import rag_backend as rag

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF RAG Chatbot")
st.caption("Upload PDFs and ask questions about them, powered by Gemini.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Your documents")
    files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if st.button("Process PDFs", use_container_width=True) and files:
        with st.spinner("Reading and indexing..."):
            total = 0
            for f in files:
                total += rag.ingest_pdf(f, f.name)
        st.success(f"Indexed {total} chunks from {len(files)} file(s).")
    st.write(f"Chunks in database: **{rag.count_chunks()}**")
    if st.button("Clear database", use_container_width=True):
        rag.clear_database()
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask something about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = rag.answer_question(question)
            except Exception as e:
                answer, sources = f"Error: {e}", []
        st.markdown(answer)
        if sources:
            st.caption("Sources: " + ", ".join(f"{s} (p.{p})" for s, p in sources))
    st.session_state.messages.append({"role": "assistant", "content": answer})
