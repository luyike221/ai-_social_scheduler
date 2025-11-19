"""
主管多智能体系统示例

本示例展示如何使用 langgraph-supervisor 创建主管多智能体系统：
1. 创建多个专业智能体（航班预订助手、酒店预订助手）
2. 创建主管智能体来协调和管理这些智能体
3. 主管根据任务自动分配给合适的智能体

单个智能体可能难以应对需要专门处理多个领域或管理多种工具的情况。为了解决这个问题，您可以将智能体分解为更小、独立的智能体，并将它们组合成一个多智能体系统。

在多智能体系统中，智能体之间需要进行通信。它们通过移交来实现这一点——这是一种描述将控制权移交给哪个智能体以及发送给该智能体的数据负载的原始操作。

两种最受欢迎的多智能体架构是：

主管——单个智能体由一个中央主管智能体协调。主管控制所有通信流和任务委派，根据当前上下文和任务要求决定调用哪个智能体。
群组——智能体根据其专业性动态地相互移交控制权。系统会记住哪个智能体上次处于活动状态，确保在后续交互中，对话会与该智能体恢复。


安装依赖：
pip install langgraph-supervisor

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.j-多智能体.supervisor_example

"""

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

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


# ==================== 创建智能体 ====================

def flight_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    """航班预订助手的提示"""
    system_msg = "你是一个航班预订助手，专门帮助用户预订航班。"
    return [{"role": "system", "content": system_msg}] + state["messages"]


def hotel_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    """酒店预订助手的提示"""
    system_msg = "你是一个酒店预订助手，专门帮助用户预订酒店。"
    return [{"role": "system", "content": system_msg}] + state["messages"]


def create_agents():
    """创建多个专业智能体"""
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    # 航班预订助手
    flight_assistant = create_react_agent(
        qwen_client.client,
        tools=[book_flight],
        prompt=flight_prompt,
        name="flight_assistant",
    )
    
    # 酒店预订助手
    hotel_assistant = create_react_agent(
        qwen_client.client,
        tools=[book_hotel],
        prompt=hotel_prompt,
        name="hotel_assistant",
    )
    
    return flight_assistant, hotel_assistant


# ==================== 创建主管 ====================

def supervisor_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    """主管智能体的提示"""
    system_msg = (
        "你管理一个酒店预订助手和一个航班预订助手。"
        "根据用户的需求，将任务分配给合适的助手。"
    )
    return [{"role": "system", "content": system_msg}] + state["messages"]


def create_supervisor_system():
    """创建主管多智能体系统"""
    try:
        from langgraph_supervisor import create_supervisor
        
        qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
        
        # 创建专业智能体
        flight_assistant, hotel_assistant = create_agents()
        
        # 创建主管智能体
        supervisor = create_supervisor(
            agents=[flight_assistant, hotel_assistant],
            model=qwen_client.client,
            prompt=supervisor_prompt,
        ).compile()
        
        return supervisor
        
    except ImportError:
        print("❌ 未安装 langgraph-supervisor")
        print("请运行: pip install langgraph-supervisor")
        return None


# ==================== 示例：使用主管系统 ====================

def example_supervisor_system():
    """示例：使用主管多智能体系统"""
    print("=" * 80)
    print("示例：主管多智能体系统")
    print("=" * 80)
    print()
    
    supervisor = create_supervisor_system()
    if supervisor is None:
        return
    
    print("💡 主管系统已创建")
    print("   智能体列表:")
    print("   - flight_assistant: 航班预订助手")
    print("   - hotel_assistant: 酒店预订助手")
    print()
    
    print("📝 运行主管系统...")
    print("   任务: 预订从北京到上海的航班，并预订一家名为 McKittrick 的酒店")
    print("-" * 80)
    
    # 使用主管系统处理任务
    for chunk in supervisor.stream(
        {
            "messages": [
                HumanMessage(content="预订从北京到上海的航班，并预订一家名为 McKittrick 的酒店")
            ]
        }
    ):
        print(chunk)
        print()
    
    print()
    print("✅ 主管系统示例完成")
    print()


# ==================== 主函数 ====================

def main():
    """运行主管多智能体系统示例"""
    try:
        example_supervisor_system()
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

