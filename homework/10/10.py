import ollama

response = ollama.chat(
    model="llama3", 
    messages=[
        {
            "role": "user",
            "content": "how=?"
        }
    ]
)

print(response["message"]["content"])
