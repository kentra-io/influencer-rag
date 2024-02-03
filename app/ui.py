import streamlit as st
from _3_run_llm_llama_cpp import process_question
from app import config
from app.vector_db.vector_db_model import VectorDbType

# Streamlit Page Configuration
st.set_page_config(
    page_title="Influencer RAG",
    page_icon="▶️",
    initial_sidebar_state="expanded",
)

vector_databases = [member.value for member in VectorDbType]

hybrid_search = "No"


def render_response(entry):
    with st.chat_message("assistant"):
        st.markdown(entry["assistant_response"])

    for title, url, score, page_content, paragraph in entry["chunks"]:
        with st.expander(f"Show chunk: {title}"):
            st.video(url)
            st.write(score)
            st.subheader("Found sentence")
            st.write(page_content)
            st.subheader("Paragraph sent to LLM as context")
            st.write(paragraph)


# Sidebar Configuration
with st.sidebar:
    st.title("▶️ Influencer RAG")
    top_k = st.slider("Top-k", min_value=1, max_value=50, value=3)
    vector_db = st.selectbox('Select Vector DB', vector_databases)
    if VectorDbType(vector_db) == VectorDbType.WEAVIATE:
        hybrid_search = st.selectbox('Use hybrid search?', ("No", "Yes"))
    if 'hybrid_search' in vars() and hybrid_search == "Yes":
        alpha = st.slider("Alpha", min_value=0.0, max_value=1.0, value=0.5)
        mmr_search = False
    else:
        alpha = config.alpha
        mmr_search = st.selectbox('MMR search?', ("No", "Yes"))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat History with Chunks
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["user_prompt"])

    render_response(entry)

# User Input and Chat Logic
if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        response = process_question(
            users_query=prompt,
            enable_vector_search=True,
            k=top_k,
            vector_db=VectorDbType(vector_db),
            hybrid_search=(hybrid_search == "Yes"),
            alpha=alpha,
            mmr_search=(mmr_search == "Yes")
        )

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
                document.metadata['paragraph'],
            )
            for document, score in response.relevant_movie_chunks
        ],
    }

    render_response(new_chat_entry)

    # Append to chat history
    st.session_state.chat_history.append(new_chat_entry)
