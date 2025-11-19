"""
在工具中同时访问配置和状态示例

本示例展示如何在工具中同时访问配置和状态。

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.h-上下文.both_context_example

"""

from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from ai_social_scheduler.client import QwenClient


@tool
def get_personalized_info(
    query: str,
    state: Annotated[AgentState, InjectedState],
    config: RunnableConfig,
) -> str:
    """获取个性化信息（同时使用配置和状态）"""
    user_id = config.get("configurable", {}).get("user_id", "unknown")
    messages = state.get("messages", [])
    message_count = len(messages)
    
    return f"用户ID: {user_id}, 查询: {query}, 当前对话消息数: {message_count}"


def main():
    """示例：在工具中同时访问配置和状态"""
    print("=" * 80)
    print("示例：在工具中同时访问配置和状态")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        qwen_client.client,
        tools=[get_personalized_info],
    )
    
    print("💡 工具同时访问配置和状态...")
    response = agent.invoke(
        {"messages": [HumanMessage(content="获取我的个性化信息")]},
        config={"configurable": {"user_id": "user_456"}}
    )
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 获取我的个性化信息")
        print(f"回答: {last_message.content}")
    
    print()
    print("✅ 同时访问配置和状态示例完成")
    print()


if __name__ == "__main__":
    main()

