import streamlit as st
from _3_run_llm_llama_cpp import ask_question

# Streamlit Page Configuration
st.set_page_config(
    page_title="Influencer RAG",
    page_icon="▶️",
    initial_sidebar_state="expanded",
)

# Sidebar Configuration
with st.sidebar:
    st.title("▶️ Influencer RAG")
    top_k = st.slider("Top-k", min_value=1, max_value=50, value=3)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat History with Chunks
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["user_prompt"])

    with st.chat_message("assistant"):
        st.markdown(entry["assistant_response"])

    for title, url, score, page_content in entry["chunks"]:
        with st.expander(f"Show chunk: {title}"):
            st.video(url)
            st.write(score)
            st.write(page_content)

# User Input and Chat Logic
if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        response = ask_question(users_query=prompt, enable_vector_search=True, k=top_k)

    # Prepare new chat entry with response and chunks
    new_chat_entry = {
        "user_prompt": prompt,
        "assistant_response": response.llm_user_response,
        "chunks": [
            (
                document.metadata["title"],
                document.metadata["url"],
                score,
                document.page_content,
            )
            for document, score in response.relevant_movie_chunks
        ],
    }

    with st.chat_message("assistant"):
        st.markdown(new_chat_entry["assistant_response"])

    for title, url, score, page_content in new_chat_entry["chunks"]:
        with st.expander(f"Show chunk: {title}"):
            st.video(url)
            st.write(score)
            st.write(page_content)

    # Append to chat history
    st.session_state.chat_history.append(new_chat_entry)
