from openai import OpenAI

def generate_text_response(prompt, text_format=None):
    client = OpenAI()
    if text_format is None:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        return response.output_text, response.usage.total_tokens
    else:
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=prompt,
            text_format=text_format,
        )
        return response.output_parsed, response.usage.total_tokens

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