from openai import OpenAI

# gpt-5.4配置
api_key = "sk-70e924fa69410befe4bad7d42a7fc18f8b07936d1a5edae43ee68bcb62985b4c"
# base_url = "https://www.ananapi.com/v1/"
base_url = "https://codex.sakurapy.de/v1/"
model = "gpt-5.4"

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "你好，简单回复一句话"}],
        temperature=0.0,
        max_tokens=2048
    )
    print("响应类型:", type(response))
    print("响应内容:", response)
except Exception as e:
    print("错误类型:", type(e))
    print("错误信息:", str(e))
    # 打印异常的完整内容
    import traceback
    traceback.print_exc()
