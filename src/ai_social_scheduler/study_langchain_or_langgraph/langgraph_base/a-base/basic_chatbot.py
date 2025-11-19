"""
构建一个带记忆功能的基本聊天机器人

本教程展示了如何使用 LangGraph 构建一个带记忆功能的基本聊天机器人。
机器人会记住整个对话历史，可以基于之前的对话内容进行回答。

主要特性：
1. 使用 LangGraph 的检查点（checkpoint）机制保存对话历史
2. 使用 InMemorySaver 在内存中保存对话状态
3. 通过 thread_id 来标识和管理不同的对话会话
4. 支持清除对话历史，开始新的对话

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.basic_chatbot

"""

import uuid
from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
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


# ==================== 2. 创建 StateGraph ====================

def create_chatbot_graph():
    """
    创建一个带记忆功能的基本聊天机器人图
    
    Returns:
        CompiledGraph: 编译后的图（支持记忆功能）
    """
    # 创建图构建器
    graph_builder = StateGraph(State)
    
    # ==================== 3. 初始化 LLM ====================
    # 使用 QwenClient（通义千问）作为 LLM
    qwen_client = QwenClient(
        model="qwen-plus",
        temperature=0.7,
    )
    llm = qwen_client.client
    
    # ==================== 4. 添加节点 ====================
    
    def chatbot(state: State):
        """
        聊天机器人节点
        
        这个节点接收当前状态作为输入，调用 LLM 生成响应，
        并返回包含新消息的状态更新。
        
        Args:
            state: 当前状态，包含消息列表（包含历史消息）
        
        Returns:
            dict: 包含新消息的状态更新
        """
        # 调用 LLM，传入当前的所有消息（包括历史消息）
        response = llm.invoke(state["messages"])
        # 返回包含 LLM 响应的状态更新
        # add_messages reducer 会将新消息追加到现有消息列表
        return {"messages": [response]}
    
    # 添加 chatbot 节点
    # 第一个参数是节点的唯一名称
    # 第二个参数是节点被调用时执行的函数或对象
    graph_builder.add_node("chatbot", chatbot)
    
    # ==================== 5. 添加入口点 ====================
    # 添加从 START 到 chatbot 的边，告诉图每次运行时从哪里开始
    graph_builder.add_edge(START, "chatbot")
    
    # ==================== 6. 创建检查点保存器（用于记忆）====================
    # InMemorySaver 将对话历史保存在内存中
    # 使用相同的 thread_id 可以访问之前的对话历史
    checkpointer = InMemorySaver()
    
    # ==================== 7. 编译图（启用检查点功能）====================
    # 在运行图之前需要编译它
    # 传入 checkpointer 参数以启用记忆功能
    graph = graph_builder.compile(checkpointer=checkpointer)
    
    return graph


# ==================== 8. 运行聊天机器人 ====================

def stream_graph_updates(graph, user_input: str, config: dict):
    """
    流式处理图更新（支持记忆功能）
    
    Args:
        graph: 编译后的图（已启用检查点）
        user_input: 用户输入
        config: 配置字典，必须包含 thread_id 来标识对话会话
    """
    # 创建用户消息
    user_message = HumanMessage(content=user_input)
    
    # 流式处理图更新
    # 传入 config 参数，使用相同的 thread_id 可以访问之前的对话历史
    for event in graph.stream({"messages": [user_message]}, config=config):
        # event 是一个字典，键是节点名称，值是该节点的输出
        for node_name, value in event.items():
            if "messages" in value and value["messages"]:
                # 打印助手的最新消息
                last_message = value["messages"][-1]
                print("助手:", last_message.content)


def run_chatbot():
    """
    运行带记忆功能的聊天机器人主循环
    
    提示：您可以随时通过键入 quit、exit 或 q 来退出聊天循环
    机器人会记住整个对话历史，可以基于之前的对话内容进行回答
    """
    print("=" * 60)
    print("基本聊天机器人（带记忆功能）- 使用 QwenClient（通义千问）")
    print("=" * 60)
    print("提示：输入 'quit'、'exit' 或 'q' 退出")
    print("提示：输入 'clear' 或 'reset' 清除对话历史")
    print("=" * 60)
    print()
    
    # 创建图（已启用记忆功能）
    graph = create_chatbot_graph()
    
    # 配置：使用 thread_id 来标识对话会话
    # 相同的 thread_id 会共享对话历史
    thread_id = "basic-chatbot-session"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 交互式聊天循环
    while True:
        try:
            user_input = input("用户: ")
            
            # 检查退出命令
            if user_input.lower() in ["quit", "exit", "q"]:
                print("再见！")
                break
            
            # 检查清除历史命令
            if user_input.lower() in ["clear", "reset"]:
                # 创建新的 thread_id 来清除历史
                thread_id = f"basic-chatbot-session-{uuid.uuid4().hex[:8]}"
                config = {"configurable": {"thread_id": thread_id}}
                print("✅ 对话历史已清除，开始新的对话")
                print()
                continue
            
            # 处理用户输入
            if user_input.strip():
                stream_graph_updates(graph, user_input, config)
                print()  # 空行分隔
            else:
                print("请输入有效的问题。")
                
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except EOFError:
            # 如果没有交互式输入可用，使用示例输入
            print("用户: 你好，请介绍一下 LangGraph")
            stream_graph_updates(graph, "你好，请介绍一下 LangGraph", config)
            print()
            print("用户: 它有什么优势？")
            stream_graph_updates(graph, "它有什么优势？", config)
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            break


# ==================== 主函数 ====================

def main():
    """主函数"""
    try:
        run_chatbot()
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

