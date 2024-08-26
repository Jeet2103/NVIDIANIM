from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama-3.1-405b-instruct",
  messages=[{"role":"user","content":"Provide me an article on Machine Learning"}],
  temperature=0.2,
  top_p=1,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

