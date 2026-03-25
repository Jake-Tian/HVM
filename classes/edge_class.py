
class Edge:
    """Edge between two nodes, supports integer IDs."""
    _id_counter = 0

    @classmethod
    def next_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    def __init__(self, timestamp, source, target, content, embedding=None):
        self.id = Edge.next_id()
        self.timestamp = timestamp
        self.source = source  
        self.target = target 
        self.content = content
        self.embedding = embedding
        
    def __repr__(self):
        return f"Edge([{self.timestamp}] {self.source} -{self.content}-> {self.target})"
