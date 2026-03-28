class Conversation:
    _id_counter = 0

    @classmethod
    def next_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    def __init__(self, clip_id, messages, speakers=None, summary=None):
        self.id = self.next_id()
        self.clips = [clip_id]
        # Messages are stored as [speaker, content, clip_id, embedding] (4 elements)
        self.messages = messages if messages else []
        self.summary = summary if summary else ""
        
        if speakers is None:
            speakers = set()
            for msg in self.messages:
                if isinstance(msg, list) and len(msg) >= 1:
                    speakers.add(msg[0])
        self.speakers = speakers if isinstance(speakers, set) else set(speakers)
    
    def add_messages(self, messages, clip_id):
        """
        Add new messages to this conversation.
        
        Args:
            messages: List of [speaker, content] pairs (2 elements)
            clip_id: ID of the clip these messages belong to
        
        Messages will be converted to [speaker, content, clip_id, embedding] format (4 elements).
        Embeddings are generated using text-embedding-3-small model.
        Deduplication is based on (speaker, content) pair.
        """
        if not messages:
            return
        
        from utils.llm import get_embedding
        
        existing_messages = set()
        for msg in self.messages:
            if isinstance(msg, list) and len(msg) >= 2:
                existing_messages.add((msg[0], msg[1]))  # Deduplicate by (speaker, content)
        
        new_messages = []
        for msg in messages:
            if isinstance(msg, list) and len(msg) >= 2:
                speaker = msg[0]
                content = msg[1]
                msg_tuple = (speaker, content)
                
                if msg_tuple not in existing_messages:
                    existing_messages.add(msg_tuple)
                    
                    # Generate embedding for the message using text-embedding-3-small
                    # Remove angle brackets from speaker name for embedding: "<Alice>" -> "Alice"
                    speaker_name = speaker
                    if speaker_name.startswith("<") and speaker_name.endswith(">"):
                        speaker_name = speaker_name[1:-1]
                    formatted_msg = f"{speaker_name}: {content}"
                    try:
                        embedding = get_embedding(formatted_msg)
                    except Exception as e:
                        print(f"Warning: Failed to get embedding for message, using None: {e}")
                        embedding = None
                    
                    # Store as [speaker, content, clip_id, embedding]
                    new_messages.append([speaker, content, clip_id, embedding])
                    self.speakers.add(speaker)
        
        if new_messages:
            self.messages.extend(new_messages)
    
    def add_clip(self, clip_id):
        if clip_id not in self.clips:
            self.clips.append(clip_id)
    
    def format_messages(self):
        """
        Transform messages into a formatted string.
        
        Converts messages from list format [['<Speaker>', 'content', clip_id, embedding], ...] 
        to string format: "Speaker: content\nSpeaker: content\n..."
        
        Returns:
            str: Formatted conversation string with speaker names (angle brackets removed)
                 and message content, separated by newlines.
        """
        formatted_lines = []
        for msg in self.messages:
            if isinstance(msg, list) and len(msg) >= 2:
                speaker = msg[0]
                content = msg[1]  # content is at index 1
                
                # Remove angle brackets from speaker name if present
                if speaker.startswith("<") and speaker.endswith(">"):
                    speaker = speaker[1:-1]
                
                formatted_lines.append(f"{speaker}: {content}")
        
        return "\n".join(formatted_lines)
    
    def __repr__(self):
        speakers_str = ", ".join(sorted(self.speakers)) if self.speakers else "none"
        clips_str = ", ".join(map(str, sorted(self.clips)))
        return f"Conversation(id={self.id}, clips=[{clips_str}], speakers=[{speakers_str}], messages={len(self.messages)})"