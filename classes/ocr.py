class OCR:
    def __init__(self, clip_id, context, content, embedding=None):
        """
        Initialize an OCR object.
        
        Args:
            clip_id (int): The ID of the clip where the information was extracted.
            context (str): A simple description of where the information was extracted (e.g., "sign in the store").
            content (str): The actual text content extracted.
            embedding (list, optional): Embedding of the content.
        """
        self.clip_id = clip_id
        self.context = context
        self.content = content
        self.embedding = embedding

    def __repr__(self):
        return f"OCR(clip_id={self.clip_id}, context='{self.context}', content='{self.content}')"
