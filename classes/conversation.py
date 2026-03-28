from utils.llm import get_multiple_embeddings

class Conversation:
    _id_counter = 0

    @classmethod
    def next_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    def __init__(self, conversation_messages, embeddings, summary=""):
        self.id = self.next_id()

        self.start_time = conversation_messages[0]["start_time"]
        self.summary = summary if summary else ""
        # [start_time, speaker, content, embedding]
        self.messages = []
        
        for i in range(len(conversation_messages)):
            self.messages.append([conversation_messages[i]["start_time"], f"<{conversation_messages[i]['speaker']}>", conversation_messages[i]["content"], embeddings[i]])
    
    def __repr__(self):
        return f"Conversation(id={self.id}, start_time={self.start_time}, summary={self.summary}, messages={len(self.messages)})"