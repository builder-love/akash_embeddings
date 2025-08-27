# chat api served by akash network
# we just need to reference the AKASH_CHAT_API_KEY from the environment variables

import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Set the API key
client = openai.OpenAI(
    api_key=os.getenv("AKASH_CHAT_API_KEY"),
    base_url="https://chatapi.akash.network/api/v1"
)

# define the context block
context_block = """

"""

# define the model's system prompt
system_prompt = f"""
### ROLE ###
You are an expert AI assistant specialized in analyzing software repositories to answer user questions.

### TASK ###
Analyze the provided software repository CONTEXT below. Your task is to answer the USER QUESTION based *only* on this context. Do not use any prior knowledge.

### RULES ###
- If the context does not contain the answer, you must state: "Based on the provided context, I cannot answer this question."
- Be concise and to the point.
- If the answer involves listing multiple repositories, use a bulleted list.
- After mentioning a repository, cite your source by including its name in parentheses, like (Repo: Etherspot/prime-sdk).

### CONTEXT ###
{context_block}
"""

# Start the conversation
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What are the account abstraction libraries in the Ethereum ecosystem?"}
]

# Generate a response
# for rapid prototyping use Meta-Llama-3-1-8B-Instruct-FP8
# for poc use Meta-Llama-3-1-405B-Instruct-FP8"
response = client.chat.completions.create(
    model="Meta-Llama-3-1-8B-Instruct-FP8",  # Specify the model
    messages=messages
)

# Print the assistant's reply
print(response.choices[0].message.content)

# Continue the conversation
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Can you tell me more about that?"})

# Generate another response
response = client.chat.completions.create(
    model="Meta-Llama-3-1-8B-Instruct-FP8",
    messages=messages
)
