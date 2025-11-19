"""
使用 Store API 的长期内存示例

本示例展示如何使用 LangGraph 的 Store API 来读写长期内存：
1. 使用 InMemoryStore 和 get_store() 读取长期内存
2. 使用 InMemoryStore 和 get_store() 写入长期内存
3. 跨会话访问长期内存

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.I-内存.store_example

"""

from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.config import get_store
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from ai_social_scheduler.client import QwenClient


# ==================== 示例 1: 从工具读取长期内存 ====================

@tool
def get_user_info(config: RunnableConfig) -> str:
    """
    查找用户信息（从长期内存读取）
    
    机制：
    - 使用 get_store() 获取 store 实例
    - store.get(("users",), user_id) 读取数据
    - 数据存储在命名空间中，可以跨会话访问
    """
    store = get_store()
    user_id = config.get("configurable", {}).get("user_id")
    
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"


def example_read_from_store():
    """示例 1: 从工具读取长期内存"""
    print("=" * 80)
    print("示例 1: 从工具读取长期内存")
    print("=" * 80)
    print()
    
    store = InMemoryStore()
    
    # 预先存储一些用户数据
    store.put(
        ("users",),
        "user_123",
        {
            "name": "张三",
            "language": "中文",
        }
    )
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        qwen_client.client,
        tools=[get_user_info],
        store=store,
    )
    
    print("💡 从长期内存读取用户信息...")
    response = agent.invoke(
        {"messages": [HumanMessage(content="查询用户信息")]},
        config={"configurable": {"user_id": "user_123"}}
    )
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 查询用户信息")
        print(f"回答: {last_message.content}")
    
    # 直接访问 store 验证数据
    user_info = store.get(("users",), "user_123")
    print(f"Store 中的数据: {user_info.value if user_info else 'None'}")
    
    print()
    print("✅ 读取长期内存示例完成")
    print()


# ==================== 示例 2: 从工具写入长期内存 ====================

class UserInfo(TypedDict):
    """用户信息类型"""
    name: str
    language: str


@tool
def save_user_info(user_info: UserInfo, config: RunnableConfig) -> str:
    """
    保存用户信息（写入长期内存）
    
    机制：
    - 使用 get_store() 获取 store 实例
    - store.put(("users",), user_id, user_info) 写入数据
    - 数据会持久化在 store 中，可以跨会话访问
    """
    store = get_store()
    user_id = config.get("configurable", {}).get("user_id")
    
    store.put(("users",), user_id, user_info)
    return f"成功保存用户信息: {user_info}"


def example_write_to_store():
    """示例 2: 从工具写入长期内存"""
    print("=" * 80)
    print("示例 2: 从工具写入长期内存")
    print("=" * 80)
    print()
    
    store = InMemoryStore()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        qwen_client.client,
        tools=[save_user_info],
        store=store,
    )
    
    print("💡 保存用户信息到长期内存...")
    response = agent.invoke(
        {"messages": [HumanMessage(content="保存我的信息：姓名是李四，语言是中文")]},
        config={"configurable": {"user_id": "user_456"}}
    )
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 保存我的信息")
        print(f"回答: {last_message.content}")
    
    # 直接访问 store 验证数据
    user_info = store.get(("users",), "user_456")
    print(f"Store 中的数据: {user_info.value if user_info else 'None'}")
    
    print()
    print("✅ 写入长期内存示例完成")
    print()


# ==================== 示例 3: 跨会话使用长期内存 ====================

@tool
def get_user_profile(config: RunnableConfig) -> str:
    """获取用户资料（从长期内存）"""
    store = get_store()
    user_id = config.get("configurable", {}).get("user_id")
    
    user_info = store.get(("users",), user_id)
    if user_info:
        info = user_info.value
        return f"用户资料 - 姓名: {info.get('name', '未知')}, 语言: {info.get('language', '未知')}"
    return "用户资料不存在"


@tool
def update_user_profile(name: str, language: str, config: RunnableConfig) -> str:
    """更新用户资料（写入长期内存）"""
    store = get_store()
    user_id = config.get("configurable", {}).get("user_id")
    
    store.put(("users",), user_id, {
        "name": name,
        "language": language
    })
    return f"成功更新用户资料: 姓名={name}, 语言={language}"


def example_cross_session_store():
    """示例 3: 跨会话使用长期内存"""
    print("=" * 80)
    print("示例 3: 跨会话使用长期内存")
    print("=" * 80)
    print()
    
    # 共享的 store（跨会话）
    shared_store = InMemoryStore()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        qwen_client.client,
        tools=[update_user_profile, get_user_profile],
        store=shared_store,
    )
    
    # 会话 1: 更新用户资料
    print("📝 会话 1: 更新用户资料...")
    response1 = agent.invoke(
        {"messages": [HumanMessage(content="更新我的资料：姓名是王五，语言是中文")]},
        config={
            "configurable": {
                "thread_id": "session_1",
                "user_id": "user_789"
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
                "user_id": "user_789"  # 相同的用户ID
            }
        }
    )
    
    if "messages" in response2 and response2["messages"]:
        last_message = response2["messages"][-1]
        print(f"回答: {last_message.content}")
    
    # 直接访问 store 验证数据
    user_info = shared_store.get(("users",), "user_789")
    print(f"Store 中的数据: {user_info.value if user_info else 'None'}")
    
    print()
    print("✅ 跨会话长期内存示例完成")
    print()


# ==================== 主函数 ====================

def main():
    """运行所有 Store API 示例"""
    try:
        # 示例 1: 从工具读取长期内存
        example_read_from_store()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 2: 从工具写入长期内存
        example_write_to_store()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 3: 跨会话使用长期内存
        example_cross_session_store()
        
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

