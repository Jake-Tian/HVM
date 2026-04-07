class BaseNode:
    """Base class for all node types with integer IDs."""
    _id_counter = 0

    @classmethod
    def next_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    def __init__(self, name):
        self.name = name
        self.id = self.next_id()
        
    @property
    def type(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.type}(id={self.id})"


class CharacterNode(BaseNode):

    def __init__(self, name, embedding=None):
        super().__init__(name)
        # Embedding will be generated in batch later via node_embedding_insertion()
        self.embedding = embedding
        self.is_main_character = False

class ObjectNode(BaseNode):
    
    def __init__(self, name, embedding=None):
        super().__init__(name)
        # Embedding will be generated in batch later via node_embedding_insertion()
        self.embedding = embedding

