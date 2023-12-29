import streamlit as st

from _3_run_llm_llama_cpp import ask_question

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        response = ask_question(users_query=prompt, enable_vector_search=True)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response.llm_user_response)

    # Display expandable chunk containers
    for chunk in response.relevant_movie_chunks:
        with st.expander("Show chunk"):
            document, score = chunk
            st.markdown("### " + document.metadata['title'])
            st.video(document.metadata['url'])
            st.write(score)
            st.write(document.page_content)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.llm_user_response
        }
    )
