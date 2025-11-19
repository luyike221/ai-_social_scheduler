"""
移交（Handoff）多智能体系统示例

本示例展示如何手动实现移交功能，不使用 langgraph-supervisor 或 langgraph-swarm：
1. 创建移交工具，允许智能体将控制权移交给其他智能体
2. 使用 Command 实现移交，指定目标智能体和传递的数据
3. 使用 StateGraph 手动构建多智能体图
4. 智能体之间可以动态地相互移交控制权

移交（Handoff）是多智能体交互中的常见模式：
- 一个智能体将控制权移交给另一个智能体
- 可以指定目标智能体和要传递的信息
- 在 langgraph-supervisor 和 langgraph-swarm 中都有使用

安装依赖：
无需额外安装（使用 LangGraph 核心功能）

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.j-多智能体.handoff_example

"""

from typing import Annotated

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command

from ai_social_scheduler.client import QwenClient


# ==================== 工具定义 ====================

@tool
def book_hotel(hotel_name: str) -> str:
    """预订酒店"""
    return f"成功预订酒店: {hotel_name}"


@tool
def book_flight(from_airport: str, to_airport: str) -> str:
    """预订航班"""
    return f"成功预订航班: 从 {from_airport} 到 {to_airport}"


# ==================== 创建移交工具 ====================

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """
    创建移交工具
    
    这个工具允许智能体将控制权移交给另一个智能体。
    使用 Command 来指定目标智能体和要传递的数据。
    
    Args:
        agent_name: 目标智能体的名称
        description: 工具描述
    
    Returns:
        移交工具
    """
    name = f"transfer_to_{agent_name}"
    description = description or f"移交给 {agent_name}"
    
    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """
        移交工具实现
        
        使用 Command 来：
        1. goto: 指定目标智能体（节点）
        2. update: 更新状态，传递消息
        3. graph=Command.PARENT: 指示 LangGraph 导航到父图中的智能体节点
        """
        tool_message = {
            "role": "tool",
            "content": f"成功移交给 {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        
        return Command(
            goto=agent_name,  # 目标智能体名称
            update={"messages": state["messages"] + [tool_message]},  # 更新消息
            graph=Command.PARENT,  # 指示导航到父图中的节点
        )
    
    return handoff_tool


# ==================== 创建智能体 ====================

def flight_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    """航班预订助手的提示"""
    system_msg = (
        "你是一个航班预订助手，专门帮助用户预订航班。"
        "重要：当你需要处理航班预订时，先完成航班预订任务，然后再考虑是否需要移交。"
        "如果需要预订酒店，在完成航班预订后，使用 transfer_to_hotel_assistant 工具将用户移交给酒店预订助手。"
        "不要同时调用多个工具，先完成当前任务再移交。"
    )
    return [{"role": "system", "content": system_msg}] + state["messages"]


def hotel_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    """酒店预订助手的提示"""
    system_msg = (
        "你是一个酒店预订助手，专门帮助用户预订酒店。"
        "重要：当你需要处理酒店预订时，先完成酒店预订任务，然后再考虑是否需要移交。"
        "如果需要预订航班，在完成酒店预订后，使用 transfer_to_flight_assistant 工具将用户移交给航班预订助手。"
        "不要同时调用多个工具，先完成当前任务再移交。"
    )
    return [{"role": "system", "content": system_msg}] + state["messages"]


def create_agents():
    """创建带有移交工具的智能体"""
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    # 创建移交工具
    # 这些工具允许智能体将控制权移交给其他智能体
    transfer_to_hotel_assistant = create_handoff_tool(
        agent_name="hotel_assistant",
        description="将用户移交给酒店预订助手。当用户需要预订酒店时使用此工具。",
    )
    
    transfer_to_flight_assistant = create_handoff_tool(
        agent_name="flight_assistant",
        description="将用户移交给航班预订助手。当用户需要预订航班时使用此工具。",
    )
    
    # 航班预订助手
    # 包含航班预订工具和移交到酒店助手的工具
    flight_assistant = create_react_agent(
        qwen_client.client,
        tools=[book_flight, transfer_to_hotel_assistant],
        prompt=flight_prompt,
        name="flight_assistant",
    )
    
    # 酒店预订助手
    # 包含酒店预订工具和移交到航班助手的工具
    hotel_assistant = create_react_agent(
        qwen_client.client,
        tools=[book_hotel, transfer_to_flight_assistant],
        prompt=hotel_prompt,
        name="hotel_assistant",
    )
    
    return flight_assistant, hotel_assistant


# ==================== 创建多智能体图 ====================

def create_multi_agent_graph():
    """创建多智能体图"""
    # 创建智能体
    flight_assistant, hotel_assistant = create_agents()
    
    # 使用 StateGraph 手动构建多智能体图
    # MessagesState 是 LangGraph 提供的消息状态类型
    multi_agent_graph = (
        StateGraph(MessagesState)
        .add_node("flight_assistant", flight_assistant)  # 添加航班助手节点
        .add_node("hotel_assistant", hotel_assistant)  # 添加酒店助手节点
        .add_edge(START, "flight_assistant")  # 从 START 开始，默认进入航班助手
        .compile()  # 编译图
    )
    
    return multi_agent_graph


# ==================== 示例：使用移交系统 ====================

def example_handoff_system():
    """示例：使用移交多智能体系统"""
    print("=" * 80)
    print("示例：移交多智能体系统")
    print("=" * 80)
    print()
    
    multi_agent_graph = create_multi_agent_graph()
    
    print("💡 移交系统已创建")
    print("   智能体列表:")
    print("   - flight_assistant: 航班预订助手（默认启动）")
    print("   - hotel_assistant: 酒店预订助手")
    print()
    print("   特点:")
    print("   - 使用 Command 实现移交")
    print("   - 智能体可以动态地将控制权移交给其他智能体")
    print("   - 手动构建 StateGraph，完全控制图结构")
    print()
    
    print("📝 运行移交系统...")
    print("   任务 1: 预订从北京到上海的航班")
    print("-" * 80)
    
    # 运行多智能体图 - 第一个任务
    for chunk in multi_agent_graph.stream(
        {
            "messages": [
                HumanMessage(content="预订从北京到上海的航班")
            ]
        }
    ):
        print(chunk)
        print()
    
    print()
    print("   任务 2: 预订一家名为 McKittrick 的酒店")
    print("-" * 80)
    
    # 运行多智能体图 - 第二个任务（演示移交功能）
    for chunk in multi_agent_graph.stream(
        {
            "messages": [
                HumanMessage(content="预订一家名为 McKittrick 的酒店")
            ]
        }
    ):
        print(chunk)
        print()
    
    print()
    print("✅ 移交系统示例完成")
    print()


# ==================== 主函数 ====================

def main():
    """运行移交多智能体系统示例"""
    try:
        example_handoff_system()
    except ValueError as e:
        print(f"\n❌ 配置错误: {e}")
        print("请确保在 .env 文件中配置了 ALIBABA_BAILIAN_API_KEY")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

