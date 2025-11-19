"""
人工在环（Human-in-the-Loop）示例

本教程展示了如何在 LangGraph 中实现人工在环控制功能。
代理可能不可靠，可能需要人工输入才能成功完成任务。
对于某些操作，您可能需要在运行前要求人工批准。

主要特性：
1. 使用 LangGraph 的检查点（checkpoint）机制保存对话历史
2. 添加人工干预工具（human_assistance），使用 interrupt 暂停执行
3. 支持暂停和恢复执行，允许人工输入和批准
4. 通过 thread_id 来标识和管理不同的对话会话

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.human_in_the_loop

"""

from typing import Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from ...client import QwenClient


# ==================== 1. 定义状态 ====================

class State(TypedDict):
    """
    状态定义
    
    messages: 消息列表
    使用 Annotated 和 add_messages 函数，新消息会被追加到列表中，而不是覆盖
    """
    messages: Annotated[list, add_messages]


# ==================== 2. 定义工具 ====================

@tool
def human_assistance(query: str) -> str:
    """
    请求人工协助的工具
    
    这个工具使用 interrupt 函数来暂停执行，等待人工输入。
    类似于 Python 的 input() 函数，但支持持久化和恢复。
    
    Args:
        query: 向人工提出的问题或请求
    
    Returns:
        str: 人工提供的响应
    """
    # ==================== interrupt() 和 Command 的配合机制 ====================
    # 
    # interrupt() 函数的工作原理：
    #
    # 第一次调用（执行被中断）：
    #   - interrupt({"query": query}) 被调用
    #   - 抛出一个 GraphInterrupt 异常
    #   - 图的执行被暂停
    #   - 状态被保存到 checkpointer
    #   - 传入的值 {"query": query} 会被保存，用于向客户端展示
    #   - 函数不会返回（因为抛出了异常）
    #
    # 恢复执行时（使用 Command）：
    #   - 节点会从开始重新执行（re-execute）
    #   - interrupt() 再次被调用
    #   - 但这次 LangGraph 检测到有 Command(resume={...})
    #   - interrupt() 不再抛出异常，而是返回 Command.resume 的值
    #   - 在这个例子中，Command(resume={"data": human_input})
    #   - 所以 interrupt() 会返回 {"data": human_input}
    #   - human_response = {"data": human_input}
    #
    # 关键点：
    #   - interrupt() 的行为取决于是否有 resume 值
    #   - 第一次：抛出异常，暂停执行
    #   - 恢复时：返回 resume 值，继续执行
    #   - 这允许节点"假装"从未被中断，正常完成执行
    # ============================================================================
    print(f"interrupt 开始: {query}")
    
    # 第一次调用：抛出异常，暂停执行
    # 恢复执行时：返回 Command(resume={...}) 中的值
    human_response = interrupt({"query": query})
    
    # 从 Command 对象中提取人工输入的数据
    # 恢复执行时，human_response 会是 Command.resume 的值
    # 在这个例子中，是 {"data": human_input}
    return human_response["data"]


# ==================== 3. 创建 StateGraph ====================

def create_human_in_the_loop_graph():
    """
    创建一个支持人工在环控制的聊天机器人图
    
    Returns:
        CompiledGraph: 编译后的图（支持记忆和工具调用，包括人工干预）
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
    tools = [human_assistance]
    
    # 将工具绑定到 LLM
    # 这告诉 LLM 可以使用哪些工具，以及如何调用它们
    llm_with_tools = llm.bind_tools(tools)
    
    # ==================== 6. 添加节点 ====================
    
    def chatbot(state: State):
        """
        聊天机器人节点
        
        这个节点接收当前状态作为输入，调用 LLM 生成响应。
        如果 LLM 决定需要使用工具，它会在响应中包含 tool_calls。
        
        注意：由于我们会在工具执行期间中断，我们限制每次最多一个工具调用，
        以避免在恢复时重复任何工具调用。
        
        Args:
            state: 当前状态，包含消息列表（包含历史消息和工具结果）
        
        Returns:
            dict: 包含新消息的状态更新
        """
        # 调用 LLM，传入当前的所有消息
        message = llm_with_tools.invoke(state["messages"])
        
        # 由于我们会在工具执行期间中断，限制每次最多一个工具调用
        # 这样可以避免在恢复时重复工具调用
        assert len(message.tool_calls) <= 1, "每次最多只能调用一个工具"
        
        return {"messages": [message]}
    
    # 添加 chatbot 节点
    graph_builder.add_node("chatbot", chatbot)
    
    # ==================== 7. 添加工具节点 ====================
    # 使用 LangGraph 预构建的 ToolNode
    # ToolNode 会自动检查消息中的 tool_calls，并调用相应的工具
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # ==================== 8. 定义边和条件边 ====================
    
    # 从 START 到 chatbot 的边
    graph_builder.add_edge(START, "chatbot")
    
    # 从 chatbot 到其他节点的条件边
    # tools_condition 会检查 chatbot 的输出中是否有 tool_calls
    # 如果有，路由到 "tools"；如果没有，路由到 END
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,  # 预构建的条件函数
        {
            "tools": "tools",  # 如果有工具调用，路由到 tools 节点
            END: END,  # 如果没有工具调用，结束
        },
    )
    
    # 从 tools 回到 chatbot 的边
    # 工具执行完成后，需要回到 chatbot 节点，让 LLM 处理工具结果并生成最终响应
    graph_builder.add_edge("tools", "chatbot")
    
    # ==================== 9. 创建检查点保存器（用于记忆和中断恢复）====================
    # MemorySaver 将对话历史和中断状态保存在内存中
    # 使用相同的 thread_id 可以访问之前的对话历史和中断状态
    # 这允许我们在任何时候恢复执行
    memory = MemorySaver()
    
    # ==================== 10. 编译图（启用检查点功能）====================
    # 在运行图之前需要编译它
    # 传入 checkpointer 参数以启用记忆功能和中断恢复
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph


# ==================== 11. 运行示例 ====================

def run_example():
    """
    运行人工在环示例
    
    演示如何使用 interrupt 暂停执行，并通过 Command 恢复执行
    """
    print("=" * 60)
    print("人工在环（Human-in-the-Loop）示例")
    print("使用 QwenClient（通义千问）")
    print("=" * 60)
    print()
    
    # 创建图（已启用记忆和工具调用功能，包括人工干预）
    graph = create_human_in_the_loop_graph()
    
    # 配置：使用 thread_id 来标识对话会话
    thread_id = "human-in-the-loop-session"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 示例查询：请求人工协助
    user_input = "我需要一些关于构建 AI 代理的专家指导。你能帮我请求人工协助吗？"
    print(f"用户: {user_input}")
    print()
    
    # 创建用户消息
    user_message = HumanMessage(content=user_input)
    
    # 流式处理图更新
    print("开始执行图...")
    print("-" * 60)
    
    # ==================== 迭代器机制详解 ====================
    # graph.stream() 返回的是一个生成器（Generator），不是列表或集合
    # 
    # 关键点：
    # 1. 生成器是"惰性"的：只有在迭代时才会执行
    # 2. 生成器是"一次性"的：一旦迭代完成，就不能再次使用
    # 3. 生成器是"实时"的：事件在图的执行过程中逐步产生
    #
    # 执行流程：
    # - 调用 graph.stream() 时，图并不会立即执行
    # - 返回一个生成器对象，等待被迭代
    # - 当 for 循环开始迭代时，图才开始执行
    # - 图的每个节点执行完成后，会 yield 一个事件
    # - for 循环会立即处理这个事件
    # - 如果图被中断（interrupt），生成器会停止产生新事件
    # - 此时 for 循环会结束，但图的状态已保存到 checkpointer
    #
    # 注意：如果图被中断，这个 events 生成器就结束了
    # 需要重新调用 graph.stream() 来获取新的生成器，继续执行
    # ========================================================
    events = graph.stream(
        {"messages": [user_message]},
        config,
        stream_mode="values",  # 使用 values 模式获取完整状态
    )
    
    # 处理事件流
    # 这个 for 循环会：
    # 1. 触发图的执行（开始迭代生成器）
    # 2. 图的每个节点执行后，会 yield 一个事件
    # 3. for 循环立即处理这个事件
    # 4. 如果图被中断，生成器停止，for 循环结束
    # 5. 如果图正常完成，生成器结束，for 循环结束
    for event in events:
        if "messages" in event and event["messages"]:
            last_message = event["messages"][-1]
            
            # 检查是否是工具调用消息
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print(f"\n[LLM 响应]")
                if hasattr(last_message, "content") and last_message.content:
                    print(f"内容: {last_message.content}")
                print(f"[工具调用] LLM 请求调用工具:")
                for tool_call in last_message.tool_calls:
                    print(f"  - 工具名称: {tool_call.get('name', 'unknown')}")
                    print(f"  - 工具参数: {tool_call.get('args', {})}")
                print()
            
            # 检查是否是工具消息
            elif isinstance(last_message, ToolMessage):
                print(f"[工具结果]")
                print(f"  工具名称: {last_message.name}")
                print(f"  工具结果: {last_message.content[:200]}...")  # 只显示前200个字符
                print()
            
            # 检查是否是最终的 AI 响应
            elif hasattr(last_message, "content") and last_message.content:
                print(f"[最终响应]")
                print(f"助手: {last_message.content}")
                print()
    
    # ==================== 检查图的状态 ====================
    # graph.get_state(config) 的作用：
    # 
    # 1. 从 checkpointer（检查点保存器）中读取图的最新状态
    # 2. 返回一个 StateSnapshot 对象，包含以下信息：
    #    - snapshot.values: 当前状态的所有值（如 messages）
    #    - snapshot.next: 下一个要执行的节点列表（如果图被中断，这里会有值）
    #    - snapshot.tasks: 待执行的任务列表（中断的任务会在这里）
    #    - snapshot.interrupts: 中断信息列表
    #    - snapshot.config: 使用的配置
    #    - snapshot.metadata: 检查点元数据
    #    - snapshot.created_at: 状态创建时间
    #
    # 3. 为什么需要检查状态？
    #    - 图的执行可能被中断（interrupt）
    #    - 如果图正常完成，snapshot.next 会是空的
    #    - 如果图被中断，snapshot.next 会包含下一个要执行的节点
    #    - 我们需要检查是否有待处理的节点，来判断是否需要恢复执行
    #
    # 4. 状态是如何保存的？
    #    - 图在执行过程中，每个节点执行后都会保存状态到 checkpointer
    #    - 如果遇到 interrupt()，状态会被保存，然后执行暂停
    #    - 通过 thread_id（在 config 中）可以访问特定会话的状态
    # ========================================================
    snapshot = graph.get_state(config)
    
    # ==================== 为什么"还有下一个节点"表示被中断？====================
    # 
    # 理解这个逻辑需要了解图的执行流程：
    #
    # 图的执行流程：
    #   START → chatbot → (检查是否有工具调用)
    #                    ↓
    #            ┌───────┴───────┐
    #            ↓               ↓
    #          tools          END (完成)
    #            ↓
    #         chatbot (处理工具结果)
    #            ↓
    #          END (完成)
    #
    # 正常完成的情况：
    #   1. chatbot 没有工具调用 → 直接到 END → snapshot.next = ()
    #   2. tools 执行完成 → 回到 chatbot → chatbot 处理完成 → END → snapshot.next = ()
    #
    # 被中断的情况：
    #   1. chatbot 调用工具 → 路由到 tools 节点
    #   2. tools 节点执行 human_assistance 工具
    #   3. human_assistance 内部调用 interrupt() → 执行暂停
    #   4. 此时 tools 节点还没有完成，但执行已被中断
    #   5. 根据图的边定义：tools → chatbot
    #   6. 所以 snapshot.next = ("chatbot",) ← 下一个要执行的节点
    #
    # 关键点：
    #   - 如果图正常完成，所有节点都执行完毕，没有下一个节点 → snapshot.next = ()
    #   - 如果图被中断，当前节点还没完成，还有待执行的节点 → snapshot.next != ()
    #   - 在这个图中，中断通常发生在 tools 节点执行时
    #   - 中断后，下一个节点通常是 "chatbot"（因为 tools → chatbot）
    #
    # 注意：
    #   - snapshot.next 可能包含多个节点（如果有并行执行）
    #   - 在这个简单图中，通常只有一个节点
    # ============================================================================
    if snapshot.next:
        print("\n" + "=" * 60)
        print("⚠️  执行已暂停（中断）")
        print("=" * 60)
        print(f"\n当前状态:")
        print(f"  - 下一个节点: {snapshot.next}")
        
        # 检查是否有待处理的任务（中断的任务）
        if snapshot.tasks:
            print(f"\n待处理的任务:")
            for task in snapshot.tasks:
                print(f"  - {task}")
        
        # 显示中断的查询信息（如果有）
        if snapshot.values and "messages" in snapshot.values:
            messages = snapshot.values["messages"]
            for msg in reversed(messages):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get("name") == "human_assistance":
                            query = tool_call.get("args", {}).get("query", "")
                            print(f"\n[中断的查询] {query}")
                            break
        
        # 提示用户输入人工响应
        print("\n" + "-" * 60)
        print("请输入人工响应（模拟专家建议）:")
        print("提示: 输入 'skip' 跳过，或直接输入响应内容")
        print("-" * 60)
        
        human_input = input("人工输入: ").strip()
        
        if human_input.lower() == "skip":
            human_input = "抱歉，暂时无法提供协助。"
        
        # ==================== Command 对象如何恢复执行 ====================
        # 
        # Command 对象是 LangGraph 提供的恢复执行机制，它的工作原理：
        #
        # 1. interrupt() 和 Command 的配合：
        #    - interrupt() 调用时会抛出一个 GraphInterrupt 异常
        #    - 这个异常会暂停图的执行，并将状态保存到 checkpointer
        #    - interrupt() 传入的值（如 {"query": query}）会被保存
        #    - 当恢复执行时，Command(resume={...}) 中的值会传递给 interrupt()
        #
        # 2. 执行流程：
        #    ┌─────────────────────────────────────────────────┐
        #    │ 第一次执行（被中断）                            │
        #    └─────────────────────────────────────────────────┘
        #    tools 节点执行
        #      ↓
        #    human_assistance 工具被调用
        #      ↓
        #    interrupt({"query": query}) 被调用
        #      ↓
        #    抛出 GraphInterrupt 异常
        #      ↓
        #    执行暂停，状态保存到 checkpointer
        #      ↓
        #    生成器结束，for 循环退出
        #
        #    ┌─────────────────────────────────────────────────┐
        #    │ 恢复执行                                        │
        #    └─────────────────────────────────────────────────┘
        #    Command(resume={"data": human_input}) 创建
        #      ↓
        #    graph.stream(human_command, config) 被调用
        #      ↓
        #    LangGraph 从 checkpointer 读取保存的状态
        #      ↓
        #    从上次中断的节点（tools）重新开始执行
        #      ↓
        #    human_assistance 工具再次执行
        #      ↓
        #    interrupt() 再次被调用，但这次：
        #      - 不再抛出异常（因为已经有 resume 值）
        #      - 直接返回 Command.resume 中的值
        #      ↓
        #    human_assistance 返回 human_input
        #      ↓
        #    tools 节点完成，继续执行后续节点
        #
        # 3. 关键机制：
        #    - interrupt() 在第一次调用时抛出异常，暂停执行
        #    - 当使用 Command(resume={...}) 恢复时，interrupt() 会返回 resume 的值
        #    - 节点会从开始重新执行（re-execute），但这次 interrupt() 会返回 resume 值
        #    - 这允许节点"假装"从未被中断，继续执行
        #
        # 4. resume 参数的结构：
        #    - 可以是单个值：Command(resume="some value")
        #    - 可以是字典：Command(resume={"data": "some value"})
        #    - 可以是中断 ID 到值的映射：Command(resume={interrupt_id: value})
        #    - 在这个例子中，我们使用字典格式 {"data": human_input}
        #      interrupt() 会返回这个字典，然后我们通过 human_response["data"] 提取值
        #
        # 5. 为什么需要 checkpointer？
        #    - interrupt 机制依赖于状态持久化
        #    - checkpointer 保存了中断时的状态和中断信息
        #    - 恢复时需要从 checkpointer 读取这些信息
        #    - 没有 checkpointer，无法恢复执行
        # ====================================================================
        print("\n恢复执行...")
        print("-" * 60)
        
        # 创建 Command 对象，指定恢复执行时传递给 interrupt() 的值
        # resume 参数的值会被传递给 interrupt() 函数，作为它的返回值
        human_command = Command(resume={"data": human_input})
        
        # ==================== 恢复执行：新的生成器 ====================
        # 注意：这里创建了一个全新的生成器！
        # 
        # 为什么需要重新调用 graph.stream()？
        # 1. 之前的 events 生成器已经结束（因为图被中断）
        # 2. 生成器是"一次性"的，不能重复使用
        # 3. 需要创建一个新的生成器来继续执行
        #
        # 执行流程：
        # - 传入 Command(resume={...}) 告诉图从上次中断的地方恢复
        # - graph.stream() 会从 checkpointer 读取保存的状态
        # - 从上次中断的节点继续执行
        # - 产生新的事件流
        # - for 循环处理这些新事件
        # ============================================================
        events = graph.stream(human_command, config, stream_mode="values")
        
        # 处理恢复执行后的事件流
        # 这是一个全新的生成器，包含恢复执行后产生的事件
        for event in events:
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                
                if isinstance(last_message, ToolMessage):
                    print(f"[工具结果] {last_message.name}: {last_message.content[:200]}...")
                    print()
                elif hasattr(last_message, "content") and last_message.content:
                    print(f"[最终响应]")
                    print(f"助手: {last_message.content}")
                    print()


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

