import os
from openai import OpenAI

def main():
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        print("DEEPSEEK_API_KEY not set")
        return
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=True
    )
    reasoning_content = ""
    content = ""
    for chunk in response:
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_content += delta.reasoning_content
        elif hasattr(delta, "content") and delta.content:
            content += delta.content
    messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": "How many Rs are there in the word 'strawberry'?"})
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=True
    )
    reasoning_content2 = ""
    content2 = ""
    for chunk in response:
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_content2 += delta.reasoning_content
        elif hasattr(delta, "content") and delta.content:
            content2 += delta.content
    print(content)
    print(content2)

if __name__ == "__main__":
    main()
