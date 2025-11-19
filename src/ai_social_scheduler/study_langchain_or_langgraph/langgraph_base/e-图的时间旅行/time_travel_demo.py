"""
图的时间旅行演示

本教程展示了如何使用 LangGraph 的时间旅行功能：
1. 回溯图的状态历史
2. 从之前的检查点恢复执行
3. 探索不同的执行路径

在典型的聊天机器人工作流程中，用户与机器人交互一次或多次以完成任务。
内存和人工干预可在图状态中启用检查点并控制未来的响应。

如果你希望用户能够从之前的响应开始，探索不同的结果怎么办？
或者，如果你希望用户能够回溯聊天机器人的工作以纠正错误或尝试不同的策略，
就像自主软件工程师等应用程序中常见的那样，该怎么办？

你可以使用 LangGraph 内置的时间旅行功能创建这类体验。

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.05-图的时间旅行.time_travel_demo
"""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from ....client import QwenClient


# ==================== 1. 定义状态 ====================

class State(TypedDict):
    """
    状态定义
    
    使用 add_messages 来管理消息列表，自动处理消息的合并和去重。
    """
    messages: Annotated[list, add_messages]


# ==================== 2. 定义搜索工具 ====================

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
        "LangGraph latest information and features": [
            {
                "url": "https://blockchain.news/news/langchain-new-features-upcoming-events-update",
                "content": "LangChain, a leading platform in the AI development space, has released its latest updates, showcasing new use cases and enhancements across its ecosystem. According to the LangChain Blog, the updates cover advancements in LangGraph Platform, LangSmith's self-improving evaluators, and revamped documentation for LangGraph."
            },
            {
                "url": "https://blog.langchain.ac.cn/langgraph-platform-announce/",
                "content": "With these learnings under our belt, we decided to couple some of our latest offerings under LangGraph Platform. LangGraph Platform today includes LangGraph Server, LangGraph Studio, plus the CLI and SDK. ... we added features in LangGraph Server to deliver on a few key value areas. Below, we'll focus on these aspects of LangGraph Platform."
            }
        ],
        "Building autonomous agents with LangGraph examples and tutorials": [
            {
                "url": "https://towardsdatascience.com/building-autonomous-multi-tool-agents-with-gemini-2-0-and-langgraph-ad3d7bd5e79d",
                "content": "Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph | by Youness Mansar | Jan, 2025 | Towards Data Science Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph A practical tutorial with full code examples for building and running multi-tool agents Towards Data Science LLMs are remarkable — they can memorize vast amounts of information, answer general knowledge questions, write code, generate stories, and even fix your grammar. In this tutorial, we are going to build a simple LLM agent that is equipped with four tools that it can use to answer a user's question."
            },
            {
                "url": "https://github.com/anmolaman20/Tools_and_Agents",
                "content": "GitHub - anmolaman20/Tools_and_Agents: This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository serves as a comprehensive guide for building AI-powered agents using Langchain and Langgraph. It provides hands-on examples, practical tutorials, and resources for developers and AI enthusiasts to master building intelligent systems and workflows."
            }
        ],
    }
    
    # 根据查询关键词匹配结果
    query_lower = query.lower()
    results = []
    
    # 简单的关键词匹配逻辑
    if "langgraph" in query_lower and ("latest" in query_lower or "information" in query_lower or "features" in query_lower):
        results = mock_results.get("LangGraph latest information and features", [])
    elif "autonomous" in query_lower and "agent" in query_lower:
        results = mock_results.get("Building autonomous agents with LangGraph examples and tutorials", [])
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


# ==================== 3. 创建 StateGraph ====================

def create_time_travel_graph():
    """
    创建一个支持时间旅行的图
    
    Returns:
        CompiledGraph: 编译后的图（支持检查点和时间旅行）
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
    tool = tavily_search_results_json
    tools = [tool]
    
    # 将工具绑定到 LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # ==================== 6. 添加节点 ====================
    
    def chatbot(state: State):
        """
        聊天机器人节点
        
        这个节点接收当前状态作为输入，调用 LLM 生成响应。
        如果 LLM 决定需要使用工具，它会在响应中包含 tool_calls。
        
        Args:
            state: 当前状态，包含消息列表
        
        Returns:
            dict: 包含新消息的状态更新
        """
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    # 添加 chatbot 节点
    graph_builder.add_node("chatbot", chatbot)
    
    # ==================== 7. 添加工具节点 ====================
    # 使用 LangGraph 预构建的 ToolNode
    tool_node = ToolNode(tools=[tool])
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
    运行时间旅行示例
    
    演示如何：
    1. 添加多个步骤（检查点）
    2. 回溯状态历史
    3. 从特定时间点恢复执行
    """
    print("=" * 80)
    print("图的时间旅行演示 - 使用 QwenClient（通义千问）")
    print("=" * 80)
    print()
    
    # 创建图
    graph = create_time_travel_graph()
    
    # 配置（使用 thread_id 来标识会话）
    config = {"configurable": {"thread_id": "1"}}
    
    # ==================== 步骤 1: 第一次交互 ====================
    print("步骤 1: 第一次交互 - 请求研究 LangGraph")
    print("-" * 80)
    
    user_input_1 = (
        "I'm learning LangGraph. "
        "Could you do some research on it for me?"
    )
    
    print(f"用户输入: {user_input_1}")
    print()
    
    # 流式处理图更新
    events = graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input_1,
                },
            ],
        },
        config,
        stream_mode="values",
    )
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
            print()
    
    # ==================== 步骤 2: 第二次交互 ====================
    print("\n步骤 2: 第二次交互 - 讨论构建自主代理")
    print("-" * 80)
    
    user_input_2 = (
        "Ya that's helpful. Maybe I'll "
        "build an autonomous agent with it!"
    )
    
    print(f"用户输入: {user_input_2}")
    print()
    
    events = graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input_2,
                },
            ],
        },
        config,
        stream_mode="values",
    )
    """
        LangGraph 的 stream_mode 主要有以下几种模式：
            "values"（当前使用的模式）
                返回完整的状态字典
                每个事件包含整个状态的快照
                适合需要访问完整状态的场景
            "updates"（默认模式，如果不指定）
                返回节点级别的更新
                每个事件只包含被更新节点的数据
                格式：{"节点名": {"更新的字段": "值"}}
                "messages"（如果支持）
                只返回消息相关的更新
    """
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
            print()
    
    # ==================== 步骤 3: 重播完整状态历史 ====================
    print("\n步骤 3: 重播完整状态历史")
    print("-" * 80)
    print("遍历所有检查点，查看图执行的完整历史：")
    print()
    
    to_replay = None
    state_count = 0
    
    for state in graph.get_state_history(config):
        state_count += 1
        num_messages = len(state.values["messages"])
        next_nodes = state.next if state.next else ()
        
        print(f"检查点 #{state_count}:")
        print(f"  - 消息数量: {num_messages}")
        print(f"  - 下一个节点: {next_nodes}")
        print(f"  - Checkpoint ID: {state.config.get('configurable', {}).get('checkpoint_id', 'N/A')}")
        print("-" * 80)
        
        # 选择一个特定的状态用于重播
        # 我们选择第二次调用中 chatbot 节点之后的状态（有 6 条消息）
        if num_messages == 6:
            to_replay = state
            print("  ⭐ 选中此状态用于重播（第二次调用中 chatbot 节点之后）")
            print()
    
    print(f"\n总共找到 {state_count} 个检查点")
    print("\n图的每一步都会保存检查点。这跨越了调用，因此你可以在完整线程的历史中回溯。")
    
    # ==================== 步骤 4: 从检查点恢复 ====================
    print("\n步骤 4: 从检查点恢复")
    print("-" * 80)
    
    if to_replay is None:
        print("⚠️  未找到合适的检查点用于重播")
        return
    
    print("从 to_replay 状态恢复，该状态位于第二次图调用中 chatbot 节点之后。")
    print("从这一点恢复将接着调用 tools 节点。")
    print()
    print(f"下一个节点: {to_replay.next}")
    print(f"配置信息: {to_replay.config}")
    print()
    
    # ==================== 步骤 5: 从某个时间点加载状态 ====================
    print("步骤 5: 从某个时间点加载状态并恢复执行")
    print("-" * 80)
    print("检查点的 to_replay.config 包含一个 checkpoint_id 时间戳。")
    print("提供此 checkpoint_id 值会告诉 LangGraph 的检查器加载该时间点的状态。")
    print()
    print("从该检查点恢复执行：")
    print("-" * 80)
    
    # 从检查点恢复执行
    # 注意：传入 None 作为输入，因为我们只是从检查点恢复，不添加新消息
    for event in graph.stream(None, to_replay.config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()
            print()
    
    print("\n✅ 图从 tools 节点恢复执行。")
    print("你可以从上面打印的第一个值是我们搜索引擎工具的响应来判断是这种情况。")
    
    # ==================== 总结 ====================
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
本教程演示了 LangGraph 的时间旅行功能：

1. 回溯图的状态历史
   - 使用 graph.get_state_history(config) 获取所有检查点
   - 每个检查点包含状态值、下一个节点、配置信息等

2. 从之前的检查点恢复执行
   - 每个检查点都有一个唯一的 checkpoint_id
   - 通过提供包含 checkpoint_id 的配置，可以从该时间点恢复执行

3. 探索不同的执行路径
   - 能够回溯和探索替代路径为调试、实验和交互式应用程序打开了无限可能
   - 可以用于纠正错误或尝试不同的策略

4. 检查点管理
   - 图的每一步都会保存检查点
   - 检查点跨越调用，可以在完整线程的历史中回溯
   - 使用 MemorySaver 作为检查点保存器（在生产环境中可以使用持久化存储）

应用场景：
- 调试复杂的多步骤工作流
- 允许用户探索不同的对话路径
- 在自主软件工程师等应用中回溯和纠正错误
- 实验和迭代不同的执行策略
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

