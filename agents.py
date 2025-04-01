from openai import OpenAI

with open('api-key.txt', 'r') as key_file:
  key = key_file.read()

client = OpenAI(
  api_key=key
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "Explain the Navier Stokes equation."}
  ]
)

print(completion.choices[0].message.content)