"""
使用状态（可变上下文）示例

本示例展示如何使用自定义状态传递可变上下文数据。

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.h-上下文.state_example

"""

from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from ai_social_scheduler.client import QwenClient


class CustomState(AgentState):
    """
    自定义状态（继承自 AgentState）
    
    原理：
    1. AgentState 是基础状态，包含 messages 字段
    2. CustomState 继承 AgentState，添加了 user_name 字段
    3. state_schema 告诉 LangGraph 状态的结构，使其能够：
       - 验证传入的状态数据
       - 在工具中通过 InjectedState 注入状态
       - 在执行过程中保持和传递状态
    """
    user_name: str


@tool
def get_user_name(
    state: Annotated[CustomState, InjectedState]
) -> str:
    """
    获取用户名（使用状态）
    
    机制说明：
    - Annotated[CustomState, InjectedState] 告诉 LangGraph：
      1. 这个参数不是工具输入（LLM 不会填充它）
      2. 应该从当前状态中注入 CustomState 类型的数据
      3. LangGraph 会自动将状态对象传递给工具
    """
    user_name = state.get("user_name", "未知用户")
    return f"当前用户名: {user_name}"


def main():
    """示例：使用状态（可变上下文）"""
    print("=" * 80)
    print("示例：使用状态（可变上下文）")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        qwen_client.client,
        tools=[get_user_name],
        state_schema=CustomState,  # 告诉 LangGraph 状态结构，使其能够处理 user_name 字段
    )
    
    print("💡 通过状态传递用户信息...")
    # 状态传递机制：
    # 1. invoke 时传入初始状态（包含 messages 和 user_name）
    # 2. LangGraph 根据 state_schema 验证和存储状态
    # 3. 在执行过程中，状态会在各个节点间传递
    # 4. 工具通过 InjectedState 可以访问状态
    # 5. 最终返回的状态包含所有字段（包括 user_name）
    response = agent.invoke({
        "messages": [HumanMessage(content="查询我的用户名")],
        "user_name": "李四"  # 自定义状态字段，会在整个执行过程中保持
    })
    print(response)
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 查询我的用户名")
        print(f"回答: {last_message.content}")
        print(f"状态中的用户名: {response.get('user_name', 'N/A')}")
    
    print()
    print("✅ 状态示例完成")
    print()


if __name__ == "__main__":
    main()

