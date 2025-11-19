"""
长期记忆（跨线程内存）示例

本示例展示如何使用长期记忆在不同会话之间存储和访问数据。

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.h-上下文.long_term_memory_example

"""

from typing import Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command

from ai_social_scheduler.client import QwenClient


# ==================== 示例 1: 从工具读取长期记忆 ====================

@tool
def get_user_preference(
    config: RunnableConfig,
) -> str:
    """
    获取用户偏好（从长期记忆读取）
    
    机制：
    - 通过 config["configurable"]["store"] 访问长期记忆
    - store 是跨线程的，可以在不同会话间共享数据
    """
    store = config.get("configurable", {}).get("store", {})
    user_id = config.get("configurable", {}).get("user_id", "unknown")
    
    # 从长期记忆中读取用户偏好
    user_prefs = store.get(f"user_prefs_{user_id}", {})
    favorite_city = user_prefs.get("favorite_city", "未设置")
    language = user_prefs.get("language", "中文")
    
    return f"用户偏好 - 喜欢的城市: {favorite_city}, 语言: {language}"


def example_read_long_term_memory():
    """示例 1: 从工具读取长期记忆"""
    print("=" * 80)
    print("示例 1: 从工具读取长期记忆")
    print("=" * 80)
    print()
    
    checkpointer = MemorySaver()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        qwen_client.client,
        tools=[get_user_preference],
        checkpointer=checkpointer,
    )
    
    # 模拟长期记忆数据（在实际应用中，这些数据会从数据库加载）
    store = {
        "user_prefs_user_123": {
            "favorite_city": "北京",
            "language": "中文"
        }
    }
    
    print("💡 从长期记忆读取用户偏好...")
    response = agent.invoke(
        {"messages": [HumanMessage(content="查询我的偏好设置")]},
        config={
            "configurable": {
                "thread_id": "session_1",
                "user_id": "user_123",
                "store": store  # 长期记忆存储
            }
        }
    )
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 查询我的偏好设置")
        print(f"回答: {last_message.content}")
    
    print()
    print("✅ 读取长期记忆示例完成")
    print()


# ==================== 示例 2: 从工具写入长期记忆 ====================

@tool
def save_user_preference(
    favorite_city: str,
    language: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """
    保存用户偏好（写入长期记忆）
    
    机制：
    - 返回 Command(update=...) 来更新状态
    - 通过 config["configurable"]["store"] 访问长期记忆
    - 更新 store 中的数据，这些数据会持久化
    """
    user_id = config.get("configurable", {}).get("user_id", "unknown")
    store = config.get("configurable", {}).get("store", {})
    
    # 更新长期记忆
    if f"user_prefs_{user_id}" not in store:
        store[f"user_prefs_{user_id}"] = {}
    
    store[f"user_prefs_{user_id}"]["favorite_city"] = favorite_city
    store[f"user_prefs_{user_id}"]["language"] = language
    
    return Command(
        update={
            "messages": [
                ToolMessage(
                    f"已保存偏好：喜欢的城市={favorite_city}, 语言={language}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


def example_write_long_term_memory():
    """示例 2: 从工具写入长期记忆"""
    print("=" * 80)
    print("示例 2: 从工具写入长期记忆")
    print("=" * 80)
    print()
    
    checkpointer = MemorySaver()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        qwen_client.client,
        tools=[save_user_preference],
        checkpointer=checkpointer,
    )
    
    # 初始化长期记忆存储
    store = {}
    
    print("💡 保存用户偏好到长期记忆...")
    response = agent.invoke(
        {"messages": [HumanMessage(content="保存我的偏好：喜欢的城市是上海，语言是中文")]},
        config={
            "configurable": {
                "thread_id": "session_2",
                "user_id": "user_456",
                "store": store  # 长期记忆存储
            }
        }
    )
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 保存我的偏好")
        print(f"回答: {last_message.content}")
        print(f"长期记忆中的数据: {store}")
    
    print()
    print("✅ 写入长期记忆示例完成")
    print()


# ==================== 示例 3: 综合示例 - 跨会话使用长期记忆 ====================

@tool
def get_user_profile(
    config: RunnableConfig,
) -> str:
    """获取用户资料（从长期记忆）"""
    store = config.get("configurable", {}).get("store", {})
    user_id = config.get("configurable", {}).get("user_id", "unknown")
    
    profile = store.get(f"user_profile_{user_id}", {})
    name = profile.get("name", "未知")
    age = profile.get("age", "未知")
    
    return f"用户资料 - 姓名: {name}, 年龄: {age}"


@tool
def update_user_profile(
    name: str,
    age: int,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """更新用户资料（写入长期记忆）"""
    user_id = config.get("configurable", {}).get("user_id", "unknown")
    store = config.get("configurable", {}).get("store", {})
    
    if f"user_profile_{user_id}" not in store:
        store[f"user_profile_{user_id}"] = {}
    
    store[f"user_profile_{user_id}"]["name"] = name
    store[f"user_profile_{user_id}"]["age"] = age
    
    return Command(
        update={
            "messages": [
                ToolMessage(
                    f"已更新用户资料：姓名={name}, 年龄={age}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


def example_cross_session_memory():
    """示例 3: 跨会话使用长期记忆"""
    print("=" * 80)
    print("示例 3: 跨会话使用长期记忆")
    print("=" * 80)
    print()
    
    checkpointer = MemorySaver()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        qwen_client.client,
        tools=[update_user_profile, get_user_profile],
        checkpointer=checkpointer,
    )
    
    # 共享的长期记忆存储（跨会话）
    shared_store = {}
    
    # 会话 1: 更新用户资料
    print("📝 会话 1: 更新用户资料...")
    response1 = agent.invoke(
        {"messages": [HumanMessage(content="更新我的资料：姓名是张三，年龄是25")]},
        config={
            "configurable": {
                "thread_id": "session_1",
                "user_id": "user_789",
                "store": shared_store
            }
        }
    )
    
    if "messages" in response1 and response1["messages"]:
        last_message = response1["messages"][-1]
        print(f"回答: {last_message.content}")
    
    print()
    
    # 会话 2: 读取用户资料（使用不同的 thread_id，但共享 store）
    print("📖 会话 2: 读取用户资料（跨会话）...")
    response2 = agent.invoke(
        {"messages": [HumanMessage(content="查询我的用户资料")]},
        config={
            "configurable": {
                "thread_id": "session_2",  # 不同的会话ID
                "user_id": "user_789",  # 相同的用户ID
                "store": shared_store  # 共享的长期记忆
            }
        }
    )
    
    if "messages" in response2 and response2["messages"]:
        last_message = response2["messages"][-1]
        print(f"回答: {last_message.content}")
        print(f"长期记忆中的数据: {shared_store}")
    
    print()
    print("✅ 跨会话长期记忆示例完成")
    print()


# ==================== 主函数 ====================

def main():
    """运行所有长期记忆示例"""
    try:
        # 示例 1: 从工具读取长期记忆
        example_read_long_term_memory()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 2: 从工具写入长期记忆
        example_write_long_term_memory()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 3: 跨会话使用长期记忆
        example_cross_session_memory()
        
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

