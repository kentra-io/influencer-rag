class Channel:
    def __init__(self, handle, id):
        self.handle = handle
        self.id = id


class RagResponse:
    def __init__(self, query, llm_user_response, llm_full_response, relevant_movie_chunks, evaluation, response_time):
        self.query = query
        self.llm_user_response = llm_user_response
        self.llm_full_response = llm_full_response
        self.relevant_movie_chunks = relevant_movie_chunks
        self.evaluation = evaluation
        self.response_time = response_time
