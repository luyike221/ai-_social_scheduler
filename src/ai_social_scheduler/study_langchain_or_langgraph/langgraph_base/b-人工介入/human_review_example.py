"""
人机协作（Human-in-the-Loop）示例

本示例展示如何在代理中审查、编辑和批准工具调用：
1. 审查工具调用 - 在工具中使用 interrupt() 暂停执行
2. 使用 Command(resume=...) 根据人工输入继续
3. 使用 add_human_in_the_loop 包装器为任何工具添加中断

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.b-人工介入.human_review_example

"""

from typing import Callable

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool as create_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

from ai_social_scheduler.client import QwenClient


# ==================== 示例 1: 在工具中使用 interrupt() ====================

@create_tool
def book_hotel(hotel_name: str) -> str:
    """
    预订酒店（需要人工审批）
    
    机制：
    - interrupt() 会暂停执行，等待人工输入
    - 人工可以接受、编辑或拒绝工具调用
    - 使用 Command(resume=...) 恢复执行
    """
    response = interrupt(
        f"尝试调用 `book_hotel`，参数: {{'hotel_name': {hotel_name}}}。"
        "请批准或建议编辑。"
    )
    
    if response["type"] == "accept":
        pass
    elif response["type"] == "edit":
        hotel_name = response["args"]["hotel_name"]
    else:
        raise ValueError(f"未知的响应类型: {response['type']}")
    
    return f"成功预订酒店: {hotel_name}"


def example_interrupt_in_tool():
    """示例 1: 在工具中使用 interrupt()"""
    print("=" * 80)
    print("示例 1: 在工具中使用 interrupt()")
    print("=" * 80)
    print()
    
    checkpointer = InMemorySaver()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        qwen_client.client,
        tools=[book_hotel],
        checkpointer=checkpointer,
    )
    
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }
    
    print("💡 运行代理，等待人工审批...")
    print("   使用 stream() 方法可以看到代理暂停在 interrupt() 处")
    print()
    print("📝 第一步：代理调用工具，等待审批")
    print("-" * 80)
    
    # 第一步：代理调用工具，会暂停在 interrupt()
    for chunk in agent.stream(
        {"messages": [HumanMessage(content="预订一家名为 McKittrick 的酒店")]},
        config
    ):
        print(chunk)
        print()
    
    print()
    print("📝 第二步：人工审批后恢复执行")
    print("-" * 80)
    print("💡 使用 Command(resume=...) 恢复执行")
    print("   可以接受: Command(resume={'type': 'accept'})")
    print("   可以编辑: Command(resume={'type': 'edit', 'args': {'hotel_name': '新酒店名'}})")
    print()
    
    # 第二步：人工审批后恢复
    print("✅ 示例：接受工具调用")
    for chunk in agent.stream(
        Command(resume={"type": "accept"}),
        config
    ):
        print(chunk)
        print()
    
    print()
    print("✅ 在工具中使用 interrupt() 示例完成")
    print()


# ==================== 示例 2: 使用 add_human_in_the_loop 包装器 ====================

def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: dict = None,
) -> BaseTool:
    """
    为任何工具添加人机协作的包装器
    
    机制：
    - 包装工具，自动添加 interrupt() 调用
    - 支持接受、编辑、响应三种操作
    - 与 Agent Inbox UI 和 Agent Chat UI 兼容
    """
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)
    
    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }
    
    @create_tool(
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        request = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": "请审查工具调用"
        }
        response = interrupt([request])[0]
        
        # 批准工具调用
        if response["type"] == "accept":
            tool_response = tool.invoke(tool_input, config)
        # 更新工具调用参数
        elif response["type"] == "edit":
            tool_input = response["args"]["args"]
            tool_response = tool.invoke(tool_input, config)
        # 用用户反馈响应 LLM
        elif response["type"] == "response":
            user_feedback = response["args"]
            tool_response = user_feedback
        else:
            raise ValueError(f"不支持的响应类型: {response['type']}")
        
        return tool_response
    
    return call_tool_with_interrupt


@create_tool
def book_hotel_simple(hotel_name: str) -> str:
    """预订酒店（简单版本，无中断）"""
    return f"成功预订酒店: {hotel_name}"


def example_wrapper_tool():
    """示例 2: 使用 add_human_in_the_loop 包装器"""
    print("=" * 80)
    print("示例 2: 使用 add_human_in_the_loop 包装器")
    print("=" * 80)
    print()
    
    checkpointer = InMemorySaver()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    # 使用包装器为工具添加人工审批
    agent = create_react_agent(
        qwen_client.client,
        tools=[
            add_human_in_the_loop(book_hotel_simple),
        ],
        checkpointer=checkpointer,
    )
    
    config = {"configurable": {"thread_id": "2"}}
    
    print("💡 使用包装器为工具添加人工审批...")
    print("   工具会自动在调用前暂停，等待人工审批")
    print()
    print("📝 第一步：代理调用工具，等待审批")
    print("-" * 80)
    
    # 第一步：代理调用工具，会暂停
    for chunk in agent.stream(
        {"messages": [HumanMessage(content="预订一家名为 McKittrick 的酒店")]},
        config
    ):
        print(chunk)
        print()
    
    print()
    print("📝 第二步：人工审批后恢复执行")
    print("-" * 80)
    print("💡 使用 Command(resume=[...]) 恢复执行")
    print("   可以接受: Command(resume=[{'type': 'accept'}])")
    print("   可以编辑: Command(resume=[{'type': 'edit', 'args': {'args': {'hotel_name': '新酒店名'}}}])")
    print()
    
    # 第二步：人工审批后恢复
    print("✅ 示例：接受工具调用")
    for chunk in agent.stream(
        Command(resume=[{"type": "accept"}]),
        config
    ):
        print(chunk)
        print()
    
    print()
    print("✅ 使用包装器示例完成")
    print()


# ==================== 主函数 ====================

def main():
    """运行所有人机协作示例"""
    try:
        # 示例 1: 在工具中使用 interrupt()
        example_interrupt_in_tool()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 2: 使用 add_human_in_the_loop 包装器
        example_wrapper_tool()
        
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

