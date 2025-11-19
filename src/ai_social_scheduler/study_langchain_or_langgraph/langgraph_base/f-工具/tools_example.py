"""
工具使用示例

本示例展示了如何在 LangGraph 中使用工具，包括：

1. 定义简单工具 - 将普通函数作为工具使用
2. 自定义工具 - 使用 @tool 装饰器
3. 使用 Pydantic 定义自定义输入模式
4. 向模型隐藏参数 - 使用 InjectedState、AgentState、RunnableConfig
5. 禁用并行工具调用
6. 直接返回工具结果 - return_direct=True
7. 强制使用工具 - tool_choice
8. 处理工具错误
9. 使用预构建工具

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.f-工具.tools_example

"""

from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field

from ai_social_scheduler.client import QwenClient


# ==================== 1. 定义简单工具 ====================

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def example_simple_tools():
    """示例 1: 定义简单工具 - 普通函数作为工具"""
    print("=" * 80)
    print("示例 1: 定义简单工具")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    # create_react_agent 自动将普通函数转换为 LangChain 工具
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[multiply]
    )
    
    response = agent.invoke({
        "messages": [HumanMessage(content="计算 5 乘以 8 的结果")]
    })
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 计算 5 乘以 8 的结果")
        print(f"回答: {last_message.content}")
    
    print()
    print("✅ 简单工具示例完成")
    print()


# ==================== 2. 自定义工具 ====================

@tool("multiply_tool", parse_docstring=True)
def multiply_tool(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a * b


def example_custom_tool():
    """示例 2: 自定义工具 - 使用 @tool 装饰器"""
    print("=" * 80)
    print("示例 2: 自定义工具")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[multiply_tool]
    )
    
    response = agent.invoke({
        "messages": [HumanMessage(content="计算 6 乘以 7 的结果")]
    })
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 计算 6 乘以 7 的结果")
        print(f"回答: {last_message.content}")
    
    print()
    print("✅ 自定义工具示例完成")
    print()


# ==================== 3. 使用 Pydantic 定义输入模式 ====================

class MultiplyInputSchema(BaseModel):
    """Multiply two numbers"""
    a: int = Field(description="First operand")
    b: int = Field(description="Second operand")


@tool("multiply_pydantic", args_schema=MultiplyInputSchema)
def multiply_pydantic(a: int, b: int) -> int:
    return a * b


def example_pydantic_schema():
    """示例 3: 使用 Pydantic 定义自定义输入模式"""
    print("=" * 80)
    print("示例 3: 使用 Pydantic 定义输入模式")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[multiply_pydantic]
    )
    
    response = agent.invoke({
        "messages": [HumanMessage(content="计算 4 乘以 9 的结果")]
    })
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 计算 4 乘以 9 的结果")
        print(f"回答: {last_message.content}")
    
    print()
    print("✅ Pydantic 输入模式示例完成")
    print()


# ==================== 4. 向模型隐藏参数 ====================

@tool
def my_tool(
    tool_arg: str,
    state: Annotated[AgentState, InjectedState],
    config: RunnableConfig,
) -> str:
    """My tool that accesses state and config.
    
    Args:
        tool_arg: 工具参数（由模型控制）
    """
    # 访问代理状态中的消息
    messages = state.get("messages", [])
    message_count = len(messages)
    
    # 访问配置信息
    config_data = config.get("configurable", {})
    
    return f"工具参数: {tool_arg}, 消息数量: {message_count}, 配置: {config_data}"


def example_hidden_parameters():
    """示例 4: 向模型隐藏参数 - 使用 InjectedState 和 RunnableConfig"""
    print("=" * 80)
    print("示例 4: 向模型隐藏参数")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[my_tool]
    )
    
    response = agent.invoke(
        {
            "messages": [HumanMessage(content="使用工具，参数是 'test'")]
        },
        config={"configurable": {"user_id": "123", "session_id": "abc"}}
    )
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 使用工具，参数是 'test'")
        print(f"回答: {last_message.content}")
    
    print()
    print("✅ 隐藏参数示例完成")
    print()


# ==================== 5. 禁用并行工具调用 ====================

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def example_disable_parallel_tool_calls():
    """示例 5: 禁用并行工具调用"""
    print("=" * 80)
    print("示例 5: 禁用并行工具调用")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0)
    
    tools = [add, multiply]
    
    # 禁用并行工具调用
    model_with_tools = qwen_client.client.bind_tools(
        tools, 
        parallel_tool_calls=False
    )
    
    agent = create_react_agent(
        model=model_with_tools,
        tools=tools
    )
    
    response = agent.invoke({
        "messages": [HumanMessage(content="计算 3 + 5 和 4 * 7 的结果")]
    })
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 计算 3 + 5 和 4 * 7 的结果")
        print(f"回答: {last_message.content}")
    
    print()
    print("✅ 禁用并行工具调用示例完成")
    print()


# ==================== 6. 直接返回工具结果 ====================

@tool(return_direct=True)
def add_direct(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def example_return_direct():
    """示例 6: 直接返回工具结果 - return_direct=True"""
    print("=" * 80)
    print("示例 6: 直接返回工具结果")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[add_direct]
    )
    
    response = agent.invoke({
        "messages": [HumanMessage(content="计算 3 + 5 的结果")]
    })
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 计算 3 + 5 的结果")
        print(f"回答: {last_message.content}")
        print("💡 注意: 工具设置了 return_direct=True，会立即返回结果并停止代理循环")
    
    print()
    print("✅ 直接返回工具结果示例完成")
    print()


# ==================== 7. 强制使用工具 ====================

@tool(return_direct=True)
def greet(user_name: str) -> str:
    """Greet user."""
    return f"Hello {user_name}!"


def example_force_tool():
    """示例 7: 强制使用工具 - tool_choice"""
    print("=" * 80)
    print("示例 7: 强制使用工具")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    tools = [greet]
    
    # 强制使用特定工具
    model_with_tools = qwen_client.client.bind_tools(
        tools, 
        tool_choice={"type": "tool", "name": "greet"}
    )
    
    agent = create_react_agent(
        model=model_with_tools,
        tools=tools
    )
    
    response = agent.invoke({
        "messages": [HumanMessage(content="Hi, I am Bob")]
    })
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: Hi, I am Bob")
        print(f"回答: {last_message.content}")
        print("💡 注意: 通过 tool_choice 强制使用了 greet 工具")
    
    print()
    print("✅ 强制使用工具示例完成")
    print()


# ==================== 8. 处理工具错误 ====================

def multiply_with_error(a: int, b: int) -> int:
    """Multiply two numbers."""
    if a == 42:
        raise ValueError("The ultimate error")
    return a * b


def example_tool_error_handling():
    """示例 8: 处理工具错误"""
    print("=" * 80)
    print("示例 8: 处理工具错误")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    # 默认情况下，代理会捕获工具错误并传递给 LLM
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[multiply_with_error]
    )
    
    print("测试正常情况:")
    response = agent.invoke({
        "messages": [HumanMessage(content="计算 5 乘以 8 的结果")]
    })
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 计算 5 乘以 8 的结果")
        print(f"回答: {last_message.content}")
    
    print()
    print("测试错误情况:")
    response = agent.invoke({
        "messages": [HumanMessage(content="计算 42 乘以 7 的结果")]
    })
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 计算 42 乘以 7 的结果")
        print(f"回答: {last_message.content}")
        print("💡 注意: 工具抛出错误时，代理会捕获并处理")
    
    print()
    print("✅ 工具错误处理示例完成")
    print()


# ==================== 9. 使用预构建工具 ====================

def example_prebuilt_tools():
    """示例 9: 使用预构建工具"""
    print("=" * 80)
    print("示例 9: 使用预构建工具")
    print("=" * 80)
    print()
    
    # 注意：预构建工具需要相应的 API Key 配置
    # 这里仅展示概念，实际使用时需要配置相应的 API Key
    
    print("💡 预构建工具示例:")
    print("1. OpenAI web_search_preview 工具（需要 OPENAI_API_KEY）")
    print("2. LangChain 集成工具（如 Tavily、SerpAPI 等）")
    print()
    print("示例代码（需要配置 API Key）:")
    print("""
    # 使用 OpenAI 预构建工具
    agent = create_react_agent(
        model="openai:gpt-4o-mini", 
        tools=[{"type": "web_search_preview"}]
    )
    
    # 使用 LangChain 集成工具
    from langchain_community.tools import TavilySearchResults
    
    tavily_tool = TavilySearchResults()
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[tavily_tool]
    )
    """)
    
    print()
    print("✅ 预构建工具示例完成")
    print()


# ==================== 主函数 ====================

def main():
    """运行所有工具示例"""
    try:
        # 示例 1: 定义简单工具
        example_simple_tools()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 2: 自定义工具
        example_custom_tool()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 3: 使用 Pydantic 定义输入模式
        example_pydantic_schema()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 4: 向模型隐藏参数
        example_hidden_parameters()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 5: 禁用并行工具调用
        example_disable_parallel_tool_calls()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 6: 直接返回工具结果
        example_return_direct()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 7: 强制使用工具
        example_force_tool()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 8: 处理工具错误
        example_tool_error_handling()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 9: 使用预构建工具
        example_prebuilt_tools()
        
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

