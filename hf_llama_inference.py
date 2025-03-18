from huggingface_hub import InferenceClient
import dotenv

dotenv.load_dotenv(".env")

client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")

# Prompting without special tokens.
prompt = "The capital of France is:"
output = client.text_generation(prompt=prompt, max_new_tokens=96)
print("Case 1 - Text generation without special tokens:")
print("Prompt:",prompt)
print("Output:",output)

prompt="<|begin_of_text|><|start_header_id|>user<|end_header_id|>What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
output = client.text_generation(
    prompt,
    max_new_tokens=100,
)

print("Case 2 - Text generation with special tokens:")
print("Prompt:",prompt)
print("Output:",output)

# Prompting using the chat method.
prompt = "The capital of France is"
output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": prompt},
    ],
    stream=False,
    max_tokens=1024,
)
print("Case 3 - Text generation using chat interface tokens:")
print("Prompt:",prompt)
print("Output:",output.choices[0].message.content)