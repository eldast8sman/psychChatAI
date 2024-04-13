import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

with open("./docs/training_data.txt", "r", encoding = "utf-8") as f:
    token_count = len(list(encoding.encode(f.read())))

print(f"The text is composed of {token_count} tokens.")
