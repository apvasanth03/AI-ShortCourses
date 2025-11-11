# Imports
import os
from openai import OpenAI

# Environment & Client Setup
fs_cloudverse_token = os.getenv("FS_CLOUDVERSE_TOKEN")
client = OpenAI(
    api_key=fs_cloudverse_token,
    base_url="https://cloudverse.freshworkscorp.com/api/v1",
)


# Helper Functions
def get_response(model: str, prompt: str) -> str:
    """
    Get response from the LLM for a given model and prompt.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
