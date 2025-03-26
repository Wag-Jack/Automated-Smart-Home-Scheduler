from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message)