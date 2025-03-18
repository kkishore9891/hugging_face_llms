from huggingface_hub import InferenceClient
import dotenv

dotenv.load_dotenv(".env")

client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")


# Obtain the system prompt.
SYSTEM_PROMPT = open("system_prompt.txt").read()


# Preparing the prompt for the LLM:
prompt=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}
<|eot_id|><|start_header_id|>user<|end_header_id|>
What's the weather in London ?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

print(prompt)

output = client.text_generation(
    prompt,
    max_new_tokens=100,
    stop=["Observation:"]
)

print(output,"\n\n\n==================================ATTEMPT 2==================================\n\n\n")

# Another format
messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London ?"},
    ]

output = client.chat.completions.create(
    messages= messages,
    stream=False,
    max_tokens=1024,
    stop=["Observation:"]
)

print(output.choices[0].message.content)