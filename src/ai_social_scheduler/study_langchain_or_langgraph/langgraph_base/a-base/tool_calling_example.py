"""
工具调用示例

本教程展示了如何在 LangGraph 中使用工具调用功能。
这是一个简化的示例，专注于演示工具调用的核心机制。

主要特性：
1. 定义工具（Mock Tavily 搜索引擎）
2. 将工具绑定到 LLM
3. 使用条件边（conditional edges）来路由工具调用
4. 支持工具调用循环：chatbot -> tools -> chatbot

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.tool_calling_example

"""

from typing import Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
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


# ==================== 2. 定义工具（Mock Tavily 搜索引擎）====================

@tool
def tavily_search(query: str, max_results: int = 2) -> dict:
    """
    Mock Tavily 网页搜索引擎工具
    
    这是一个模拟的 Tavily 搜索引擎，返回模拟的搜索结果。
    在实际应用中，您可以使用真实的 Tavily API。
    
    Args:
        query: 搜索查询字符串
        max_results: 返回的最大结果数量（默认 2）
    
    Returns:
        dict: 包含搜索结果的字典
    """
    # Mock 搜索结果数据
    mock_results = {
        "LangGraph": [
            {
                "title": "LangGraph 简介：初学者指南 - Medium",
                "url": "https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide",
                "content": "LangGraph 是一个用于构建状态化、多参与者 LLM 应用的库。它围绕状态化图的概念展开，其中图中的每个节点代表计算的一个步骤，图维护一个状态，在计算过程中传递和更新。LangGraph 支持条件边，允许您根据图的当前状态动态确定要执行的下一个节点。",
                "score": 0.85,
            },
            {
                "title": "LangGraph 教程：什么是 LangGraph 以及如何使用？",
                "url": "https://www.datacamp.com/tutorial/langgraph-tutorial",
                "content": "LangGraph 是 LangChain 生态系统中的一个库，提供了一个框架，用于以结构化和高效的方式定义、协调和执行多个 LLM 代理（或链）。通过管理数据流和操作序列，LangGraph 允许开发人员专注于应用程序的高级逻辑，而不是代理协调的复杂性。",
                "score": 0.75,
            },
        ],
        "Python": [
            {
                "title": "Python 官方文档",
                "url": "https://www.python.org/doc/",
                "content": "Python 是一种高级、解释型的编程语言，具有清晰的语法和强大的功能。它支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。",
                "score": 0.90,
            },
            {
                "title": "Python 教程 - 菜鸟教程",
                "url": "https://www.runoob.com/python/python-tutorial.html",
                "content": "Python 是一种解释型、面向对象、动态数据类型的高级程序设计语言。Python 由 Guido van Rossum 于 1989 年底发明，第一个公开发行版发行于 1991 年。",
                "score": 0.80,
            },
        ],
        "AI": [
            {
                "title": "人工智能概述 - Wikipedia",
                "url": "https://zh.wikipedia.org/wiki/人工智能",
                "content": "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
                "score": 0.88,
            },
            {
                "title": "机器学习基础",
                "url": "https://www.example.com/ml-basics",
                "content": "机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习和改进。",
                "score": 0.82,
            },
        ],
    }
    
    # 根据查询关键词匹配结果
    query_lower = query.lower()
    results = []
    
    # 简单的关键词匹配逻辑
    if "langgraph" in query_lower or "lang graph" in query_lower:
        results = mock_results.get("LangGraph", [])[:max_results]
    elif "python" in query_lower:
        results = mock_results.get("Python", [])[:max_results]
    elif "ai" in query_lower or "人工智能" in query or "机器学习" in query:
        results = mock_results.get("AI", [])[:max_results]
    else:
        # 默认返回通用结果
        results = [
            {
                "title": f"关于 '{query}' 的搜索结果",
                "url": "https://www.example.com/search",
                "content": f"这是关于 '{query}' 的模拟搜索结果。在实际应用中，这里会返回真实的网页搜索结果。",
                "score": 0.70,
            }
        ]
    
    # 构建返回结果
    return {
        "query": query,
        "follow_up_questions": None,
        "answer": None,
        "images": [],
        "results": results,
        "response_time": 0.5,  # Mock 响应时间
    }


# ==================== 3. 创建 StateGraph ====================

def create_tool_calling_graph():
    """
    创建一个工具调用示例图
    
    Returns:
        CompiledGraph: 编译后的图（支持工具调用）
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
    tools = [tavily_search]
    
    # 将工具绑定到 LLM
    # 这告诉 LLM 可以使用哪些工具，以及如何调用它们
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
        # 调用 LLM，传入当前的所有消息
        print("调用工具开始")
        response = llm_with_tools.invoke(state["messages"])

        print("调用工具完成")
        # 返回包含 LLM 响应的状态更新
        return {"messages": [response]}
    
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
        # 映射条件输出到节点名称
        {
            "tools": "tools",  # 如果有工具调用，路由到 tools 节点
            END: END,  # 如果没有工具调用，结束
        },
    )
    
    # 从 tools 回到 chatbot 的边
    # 工具执行完成后，需要回到 chatbot 节点，让 LLM 处理工具结果并生成最终响应
    graph_builder.add_edge("tools", "chatbot")
    
    # ==================== 9. 编译图 ====================
    # 编译图（不启用检查点，因为这是简化示例）
    graph = graph_builder.compile()
    
    return graph


# ==================== 10. 运行示例 ====================

def run_example():
    """
    运行工具调用示例
    
    演示如何使用工具调用功能
    """
    print("=" * 60)
    print("工具调用示例 - 使用 QwenClient（通义千问）")
    print("=" * 60)
    print()
    
    # 创建图
    graph = create_tool_calling_graph()
    
    # 示例查询
    user_input = "请介绍一下 LangGraph"
    print(f"用户: {user_input}")
    print()
    
    # 创建用户消息
    user_message = HumanMessage(content=user_input)
    
    # 流式处理图更新
    for event in graph.stream({"messages": [user_message]}):
        # event 是一个字典，键是节点名称，值是该节点的输出
        for node_name, value in event.items():
            if "messages" in value and value["messages"]:
                last_message = value["messages"][-1]
                
                # 检查是否是工具调用消息
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    # 显示工具调用信息
                    print(f"[工具调用] {node_name} 节点正在调用工具...")
                    for tool_call in last_message.tool_calls:
                        print(f"  - 工具: {tool_call.get('name', 'unknown')}")
                        print(f"  - 参数: {tool_call.get('args', {})}")
                    print()
                elif isinstance(last_message, ToolMessage):
                    # 工具执行结果
                    print(f"[工具结果] {node_name} 节点执行完成")
                    print()
                else:
                    # 打印助手的最新消息
                    if hasattr(last_message, "content") and last_message.content:
                        print("助手:", last_message.content)
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

