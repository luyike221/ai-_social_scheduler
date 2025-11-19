"""
使用配置（静态上下文）示例

本示例展示如何使用 config 传递静态上下文数据。

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.h-上下文.config_example

"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ai_social_scheduler.client import QwenClient


@tool
def get_user_info(config: RunnableConfig) -> str:
    """查找用户信息（使用配置）"""
    user_id = config.get("configurable", {}).get("user_id", "unknown")
    user_name = config.get("configurable", {}).get("user_name", "Guest")
    return f"用户ID: {user_id}, 用户名: {user_name}"


def main():
    """示例：使用配置（静态上下文）"""
    print("=" * 80)
    print("示例：使用配置（静态上下文）")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        qwen_client.client,
        tools=[get_user_info],
    )
    
    print("💡 通过 config 传递用户信息...")
    response = agent.invoke(
        {"messages": [HumanMessage(content="查找用户信息")]},
        config={"configurable": {"user_id": "user_123", "user_name": "张三"}}
    )
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 查找用户信息")
        print(f"回答: {last_message.content}")
    
    print()
    print("✅ 配置示例完成")
    print()


if __name__ == "__main__":
    main()

