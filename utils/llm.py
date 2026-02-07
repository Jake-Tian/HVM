from openai import OpenAI
from typing import Optional

from utils.token_monitor import TokenMonitor


def generate_text_response(prompt, token_monitor: Optional[TokenMonitor] = None):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI API returned None content. The response may have been filtered or empty.")
    if token_monitor is not None and response.usage:
        token_monitor.add_text_usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
    return content

def get_embedding(text):
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text, 
    )
    return response.data[0].embedding

def get_multiple_embeddings(texts):
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts, 
    )
    return [response.data[i].embedding for i in range(len(response.data))]