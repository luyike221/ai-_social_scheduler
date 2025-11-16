"""通义千问客户端同步测试"""

from langchain_core.messages import HumanMessage, SystemMessage

from ai_social_scheduler.client import QwenClient


def test_qwen_client():
    """测试通义千问客户端（同步）"""
    print("=" * 60)
    print("通义千问客户端测试（同步）")
    print("=" * 60)

    # 使用默认配置（从 .env 读取）
    client = QwenClient()

    # 可以自定义配置
    # client = QwenClient(
    #     model="qwen-turbo",
    #     temperature=0.8,
    #     max_tokens=1000,
    # )

    messages = [
        SystemMessage(content="你是一个有用的AI助手。"),
        HumanMessage(content="请用一句话介绍通义千问。"),
    ]

    print(f"模型: {client.model}")
    print(f"温度: {client.temperature}")
    print(f"消息: {messages[-1].content}")
    print("-" * 60)

    # 同步生成文本
    print("\n生成文本:")
    response = client.generate_sync(messages)
    print(f"回复: {response}")

    # 同步流式生成
    print("\n流式生成:")
    for chunk in client.generate_stream_sync(messages):
        print(chunk, end="", flush=True)
    print("\n" + "=" * 60)


def test_dynamic_config():
    """测试动态配置"""
    print("\n" + "=" * 60)
    print("动态配置测试")
    print("=" * 60)

    client = QwenClient(model="qwen-plus", temperature=0.7)

    print(f"初始配置 - 模型: {client.model}, 温度: {client.temperature}")

    # 更新配置
    client.update_config(temperature=0.9, model="qwen-turbo")

    print(f"更新后 - 模型: {client.model}, 温度: {client.temperature}")
    print("=" * 60)


def main():
    """主函数"""
    try:
        test_qwen_client()
        test_dynamic_config()
        print("\n✅ 测试完成！")
    except ValueError as e:
        print(f"\n❌ 配置错误: {e}")
        print("请确保在 .env 文件中配置了 ALIBABA_BAILIAN_API_KEY")
    except Exception as e:
        print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()

