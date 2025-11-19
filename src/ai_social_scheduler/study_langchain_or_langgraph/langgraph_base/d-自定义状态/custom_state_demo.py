"""
自定义状态演示

本教程展示了如何在 LangGraph 中向状态添加额外的字段，以定义复杂的行为。
聊天机器人将使用搜索工具查找特定信息，并将其转发给人工进行审查。

主要特性：
1. 向状态添加自定义键（name 和 birthday）
2. 在工具内部更新状态（使用 Command）
3. 使用 interrupt 进行人工干预
4. 手动更新状态（使用 graph.update_state）

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.04-自定义状态.custom_state_demo
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from ....client import QwenClient


# ==================== 1. 向状态添加键 ====================

class State(TypedDict):
    """
    状态定义
    
    通过向状态添加 name 和 birthday 键来更新聊天机器人，以研究实体的生日。
    这些信息使其他图节点（例如存储或处理信息的下游节点）以及图的持久层易于访问。
    """
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


# ==================== 2. 定义工具 ====================

@tool
def tavily_search_results_json(query: str) -> str:
    """
    Mock Tavily 网页搜索引擎工具
    
    这是一个模拟的 Tavily 搜索引擎，返回模拟的搜索结果。
    在实际应用中，您可以使用真实的 Tavily API。
    
    Args:
        query: 搜索查询字符串
    
    Returns:
        str: JSON 格式的搜索结果字符串
    """
    import json
    
    # Mock 搜索结果数据
    mock_results = {
        "LangGraph release date": [
            {
                "url": "https://blog.langchain.ac.cn/langgraph-cloud/",
                "content": "We also have a new stable release of LangGraph. By LangChain 6 min read Jun 27, 2024 (Oct '24) Edit: Since the launch of LangGraph Platform, we now have multiple deployment options alongside LangGraph Studio - which now fall under LangGraph Platform. LangGraph Platform is synonymous with our Cloud SaaS deployment option."
            },
            {
                "url": "https://changelog.langchain.ac.cn/announcements/langgraph-cloud-deploy-at-scale-monitor-carefully-iterate-boldly",
                "content": "LangChain - Changelog | ☁ 🚀 LangGraph Platform: Deploy at scale, monitor LangChain LangSmith LangGraph LangChain LangSmith LangGraph LangChain LangSmith LangGraph LangChain Changelog Sign up for our newsletter to stay up to date DATE: The LangChain Team LangGraph LangGraph Platform ☁ 🚀 LangGraph Platform: Deploy at scale, monitor carefully, iterate boldly DATE: June 27, 2024 AUTHOR: The LangChain Team LangGraph Platform is now in closed beta, offering scalable, fault-tolerant deployment for LangGraph agents. LangGraph Platform also includes a new playground-like studio for debugging agent failure modes and quick iteration: Join the waitlist today for LangGraph Platform. And to learn more, read our blog post announcement or check out our docs. Subscribe By clicking subscribe, you accept our privacy policy and terms and conditions."
            }
        ],
    }
    
    # 根据查询关键词匹配结果
    query_lower = query.lower()
    results = []
    
    # 简单的关键词匹配逻辑
    if "langgraph" in query_lower and ("release" in query_lower or "date" in query_lower):
        results = mock_results.get("LangGraph release date", [])
    else:
        # 默认返回通用结果
        results = [
            {
                "url": "https://www.example.com/search",
                "content": f"这是关于 '{query}' 的模拟搜索结果。在实际应用中，这里会返回真实的网页搜索结果。"
            }
        ]
    
    # 返回 JSON 格式的字符串
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def human_assistance(
    name: str, 
    birthday: str, 
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """
    请求人工协助
    
    这个工具会中断执行，等待人工审查信息。
    如果信息正确，则更新状态；否则，接收人工提供的修正信息。
    
    注意：因为我们要生成一个用于状态更新的 ToolMessage，
    所以通常需要对应工具调用的 ID。我们可以使用 LangChain 的
    InjectedToolCallId 来标记这个参数不应该在工具的模式中向模型显示。
    
    Args:
        name: 实体名称
        birthday: 生日信息
        tool_call_id: 工具调用 ID（由 LangChain 自动注入）
    
    Returns:
        Command: 包含状态更新的 Command 对象
    """
    # 中断执行，等待人工响应
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    
    # 如果信息正确，按原样更新状态
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # 否则，接收人工审查者提供的信息
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"
    
    # 这次我们在工具内部显式地使用 ToolMessage 更新状态
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    
    # 我们在工具中返回一个 Command 对象来更新我们的状态
    return Command(update=state_update)


# ==================== 3. 创建 StateGraph ====================

def create_custom_state_graph():
    """
    创建一个自定义状态的图
    
    Returns:
        CompiledGraph: 编译后的图（支持自定义状态和人工干预）
    """
    # 创建图构建器
    graph_builder = StateGraph(State)
    
    # ==================== 4. 初始化 LLM ====================
    # 使用 QwenClient（通义千问）作为 LLM
    qwen_client = QwenClient(
        model="qwen-plus",
        temperature=0.7,
    )
    llm = qwen_client.client
    
    # ==================== 5. 准备工具 ====================
    tools = [tavily_search_results_json, human_assistance]
    
    # 将工具绑定到 LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # ==================== 6. 添加节点 ====================
    
    def chatbot(state: State):
        """
        聊天机器人节点
        
        这个节点接收当前状态作为输入，调用 LLM 生成响应。
        如果 LLM 决定需要使用工具，它会在响应中包含 tool_calls。
        
        Args:
            state: 当前状态，包含消息列表、name 和 birthday
        
        Returns:
            dict: 包含新消息的状态更新
        """
        # 调用 LLM，传入当前的所有消息
        message = llm_with_tools.invoke(state["messages"])
        # 确保每次只有一个工具调用
        assert len(message.tool_calls) <= 1
        return {"messages": [message]}
    
    # 添加 chatbot 节点
    graph_builder.add_node("chatbot", chatbot)
    
    # ==================== 7. 添加工具节点 ====================
    # 使用 LangGraph 预构建的 ToolNode
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # ==================== 8. 定义边和条件边 ====================
    
    # 从 START 到 chatbot 的边
    graph_builder.add_edge(START, "chatbot")
    
    # 从 chatbot 到其他节点的条件边
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    
    # 从 tools 回到 chatbot 的边
    graph_builder.add_edge("tools", "chatbot")
    
    # ==================== 9. 编译图（启用检查点）====================
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph


# ==================== 10. 运行示例 ====================

def run_example():
    """
    运行自定义状态示例
    
    演示如何使用自定义状态和人工干预功能
    """
    print("=" * 80)
    print("自定义状态演示 - 使用 QwenClient（通义千问）")
    print("=" * 80)
    print()
    
    # 创建图
    graph = create_custom_state_graph()
    
    # 配置（使用 thread_id 来标识会话）
    config = {"configurable": {"thread_id": "1"}}
    
    # ==================== 步骤 1: 提示聊天机器人 ====================
    print("步骤 1: 提示聊天机器人查找 LangGraph 的发布日期")
    print("-" * 80)
    
    user_input = (
        "Can you look up when LangGraph was released? "
        "When you have the answer, use the human_assistance tool for review."
    )
    
    print(f"用户输入: {user_input}")
    print()
    
    # 流式处理图更新
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
            print()
    
    # ==================== 步骤 2: 检查状态（应该被中断）====================
    print("\n步骤 2: 检查图的状态（应该被中断）")
    print("-" * 80)
    
    snapshot = graph.get_state(config)
    
    print(f"状态快照信息:")
    print(f"  - snapshot.next: {snapshot.next}")
    print(f"  - 是否被中断: {'是' if snapshot.next else '否'}")
    
    if snapshot.values:
        print(f"  - 当前 name: {snapshot.values.get('name', 'N/A')}")
        print(f"  - 当前 birthday: {snapshot.values.get('birthday', 'N/A')}")
    
    # ==================== 步骤 3: 添加人工协助 ====================
    print("\n步骤 3: 添加人工协助（提供正确的信息）")
    print("-" * 80)
    
    if not snapshot.next:
        print("⚠️  图没有被中断，跳过人工干预步骤")
        print()
    else:
        print("⚠️  图被中断，需要人工干预")
        print("\n聊天机器人未能识别正确的日期，因此为其提供信息")
        print("（在实际应用中，这里会等待真实的人工输入）")
        print()
        
        # 聊天机器人未能识别正确的日期，因此为其提供信息
        # 注意：resume 中的值会被传递给 interrupt() 函数作为返回值
        # 如果 resume 中包含 "correct" 字段且以 'y' 开头，则使用原始值
        # 否则，使用 resume 中的 "name" 和 "birthday" 值
        human_command = Command(
            resume={
                "name": "LangGraph",
                "birthday": "Jan 17, 2024",
                # 注意：这里没有 "correct" 字段，所以会使用 resume 中的值
            },
        )
        
        print("人工提供信息:")
        print(f"  - name: LangGraph")
        print(f"  - birthday: Jan 17, 2024")
        print()
        
        events = graph.stream(human_command, config, stream_mode="values")
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
                print()
    
    # ==================== 步骤 4: 查看状态中的字段 ====================
    print("\n步骤 4: 查看状态中的字段")
    print("-" * 80)
    
    snapshot = graph.get_state(config)
    
    # 只显示 name 和 birthday 字段
    state_fields = {k: v for k, v in snapshot.values.items() if k in ("name", "birthday")}
    print(f"状态中的自定义字段:")
    for key, value in state_fields.items():
        print(f"  - {key}: {value}")
    
    # ==================== 步骤 5: 手动更新状态 ====================
    print("\n步骤 5: 手动更新状态（使用 graph.update_state）")
    print("-" * 80)
    
    # LangGraph 对应用程序状态提供高度控制。
    # 例如，在任何时候（包括中断时），您都可以使用 graph.update_state 手动覆盖一个键。
    result = graph.update_state(config, {"name": "LangGraph (library)"})
    print(f"状态更新结果: {result}")
    
    # ==================== 步骤 6: 查看新值 ====================
    print("\n步骤 6: 查看更新后的状态值")
    print("-" * 80)
    
    snapshot = graph.get_state(config)
    state_fields = {k: v for k, v in snapshot.values.items() if k in ("name", "birthday")}
    print(f"更新后的状态字段:")
    for key, value in state_fields.items():
        print(f"  - {key}: {value}")
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
本教程演示了：

1. 向状态添加自定义键（name 和 birthday）
   - 使其他图节点和持久层易于访问这些信息

2. 在工具内部更新状态
   - 使用 Command 对象从工具内部发出状态更新
   - 使用 ToolMessage 来更新消息列表

3. 使用 interrupt 进行人工干预
   - 在工具中调用 interrupt() 来暂停执行
   - 使用 Command(resume={...}) 来恢复执行并提供数据

4. 手动更新状态
   - 使用 graph.update_state() 在任何时候手动覆盖状态键
   - 手动状态更新会在 LangSmith 中生成追踪

5. 状态管理
   - 使用 graph.get_state() 获取状态快照
   - 状态快照包含 values、next、tasks、interrupts 等信息
    """)


# ==================== 主函数 ====================

def main():
    """主函数"""
    try:
        run_example()
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

