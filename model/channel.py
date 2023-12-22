class Channel:
    def __init__(self, handle, id):
        self.handle = handle
        self.id = id


class RagResponse:
    def __init__(self, query, context, answer, evaluation, response_time):
        self.query = query
        self.context = context,
        self.answer = answer,
        self.evaluation = evaluation,
        self.response_time = response_time
