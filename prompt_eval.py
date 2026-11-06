# -*- coding: utf-8 -*-
"""
Prompt 调优对比脚本
功能：同一个任务，用 3 种不同的提示词写法调用大模型，直观对比输出差异。
运行：python prompt_eval.py
说明：这是简历上「Prompt 调优实践」的可运行证据。
      脚本把三种 prompt 写法的模型输出并排展示，效果差异一目了然，便于讲解。
"""
import os
from openai import OpenAI

GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
CHAT_MODEL = "glm-4-flash"


def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


# 同一个任务，三版 prompt
TASK = "请介绍「栈」这种数据结构。"

PROMPTS = {
    # 版本 A：裸问，没有任何约束
    "A-基础提问": TASK,

    # 版本 B：加了角色设定
    "B-加角色设定": (
        "你是一位计算机专业课老师，正在给大一学生讲解数据结构。\n"
        + TASK
    ),

    # 版本 C：角色设定 + 示例 + 限定输出格式
    "C-角色+示例+格式": (
        "你是一位计算机专业课老师，正在给大一学生讲解数据结构。\n"
        + TASK
        + "\n请按以下格式回答，共 4 个部分：\n"
        "1. 定义（一句话）\n"
        "2. 核心特点（分条列出 3 点）\n"
        "3. 常见应用（举 2 个例子）\n"
        "4. 和队列的区别（一句话）\n"
        "示例（用「队列」演示格式）：\n"
        "1. 定义：先进先出的线性表。\n"
        "2. 特点：只允许在一端插入、另一端删除……\n"
    ),
}


def call_model(client, prompt):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def main():
    load_env()
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("错误：未配置 ZHIPU_API_KEY，请在 .env 文件中填写")
        return

    client = OpenAI(api_key=api_key, base_url=GLM_BASE_URL)

    print("=" * 60)
    print("Prompt 调优对比：同一个任务，三种提示词写法")
    print("=" * 60)
    print(f"\n任务：{TASK}\n")

    for name, prompt in PROMPTS.items():
        print("\n" + "─" * 60)
        print(f"【{name}】")
        print(f"提示词：{prompt[:80]}...")
        answer = call_model(client, prompt)
        print(f"\n模型输出：\n{answer}\n")

    print("\n" + "=" * 60)
    print("对比要点（面试时这样讲）")
    print("=" * 60)
    print("""
- A 基础提问：模型自由发挥，输出往往冗长、结构松散，格式不可控。
- B 加角色设定：语气更像老师了，但输出结构依然不稳定。
- C 角色+示例+格式：因为给了「格式模板 + 示例」，模型会严格照模板输出，
  结构稳定、干净利落。

结论：真正让输出变稳定的是「给示例 + 限定格式」，角色设定主要影响语气和视角。
这就是我在实践中总结出的 Prompt 调优经验。
""")

    print("（提示：多跑几次，A、B 两版的输出每次都会有波动，C 版最稳定，这正是关键差别。）")


if __name__ == "__main__":
    main()
