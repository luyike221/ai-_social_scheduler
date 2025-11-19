"""
流式传输示例

流式传输是构建响应式应用程序的关键。本示例展示了如何在 LangGraph 中实现以下流式传输功能：

1. 代理进度流式传输 - 在每个节点执行后获取更新（stream_mode="updates"）
2. LLM 令牌流式传输 - 在语言模型生成令牌时进行流式传输（stream_mode="messages"）
3. 工具更新流式传输 - 从工具发出自定义数据（使用 get_stream_writer）
4. 多种模式同时流式传输 - 同时流式传输多种类型的数据
5. 禁用流式传输 - 在多代理系统中控制哪些代理流式传输其输出

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.a-base.streaming_example

"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.prebuilt import create_react_agent

from ai_social_scheduler.client import QwenClient


# ==================== 工具定义 ====================

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    
    这是一个示例工具，演示如何在工具中使用 get_stream_writer 进行自定义流式传输。
    
    Args:
        city: 城市名称
    
    Returns:
        str: 天气信息
    """
    # 获取流式写入器，用于发送自定义更新
    writer = get_stream_writer()
    
    # 流式传输任意自定义数据
    if writer:
        writer(f"正在查找城市数据: {city}")
        writer(f"正在查询 {city} 的天气信息...")
        writer(f"已获取 {city} 的天气数据")
    
    # 模拟天气数据
    weather_data = {
        "北京": "晴朗，温度 25°C，湿度 60%",
        "上海": "多云，温度 22°C，湿度 70%",
        "深圳": "晴朗，温度 28°C，湿度 65%",
        "广州": "小雨，温度 20°C，湿度 80%",
    }
    
    return f"{city} 的天气：{weather_data.get(city, '晴朗，温度 25°C')}"


# ==================== 1. 代理进度流式传输 ====================

def example_agent_progress_streaming():
    """
    示例 1: 代理进度流式传输
    
    使用 stream_mode="updates" 在每个代理步骤后获取更新。
    例如，如果代理调用工具一次，您应该会看到以下更新：
    - LLM 节点：带有工具调用请求的 AI 消息
    - 工具节点：带有执行结果的工具消息
    - LLM 节点：最终 AI 响应
    """
    print("=" * 80)
    print("示例 1: 代理进度流式传输 (stream_mode='updates')")
    print("=" * 80)
    print()
    
    # 创建 QwenClient
    qwen_client = QwenClient(
        model="qwen-plus",
        temperature=0.7,
    )
    
    # 创建代理
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[get_weather],
    )
    
    # 流式传输代理进度
    print("开始流式传输代理进度...")
    print("-" * 80)
    
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]},
        stream_mode="updates"
    ):
        print(f"📦 更新块:")
        print(f"   {chunk}")
        print()
    
    print("✅ 代理进度流式传输完成")
    print()


# ==================== 2. LLM 令牌流式传输 ====================

def example_llm_token_streaming():
    """
    示例 2: LLM 令牌流式传输
    
    使用 stream_mode="messages" 在语言模型生成令牌时进行流式传输。
    """
    print("=" * 80)
    print("示例 2: LLM 令牌流式传输 (stream_mode='messages')")
    print("=" * 80)
    print()
    
    # 创建 QwenClient
    qwen_client = QwenClient(
        model="qwen-plus",
        temperature=0.7,
    )
    
    # 创建代理
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[get_weather],
    )
    
    # 流式传输 LLM 令牌
    print("开始流式传输 LLM 令牌...")
    print("-" * 80)
    print("实时输出: ", end="", flush=True)
    
    try:
        # 尝试解包为 (token, metadata) 元组
        for item in agent.stream(
            {"messages": [{"role": "user", "content": "请用一句话介绍北京"}]},
            stream_mode="messages"
        ):
            # 根据 LangGraph 的实际 API，返回格式可能是元组或消息对象
            if isinstance(item, tuple) and len(item) == 2:
                token, metadata = item
                if token:
                    print(token, end="", flush=True)
            elif hasattr(item, 'content'):
                # 如果是消息对象，直接获取内容
                if item.content:
                    print(item.content, end="", flush=True)
            else:
                # 其他格式，尝试直接打印
                print(item, end="", flush=True)
    except Exception as e:
        print(f"\n⚠️  流式传输过程中出现错误: {e}")
        print("提示: 某些 LLM 客户端可能不支持 stream_mode='messages'")
    
    print()  # 换行
    print()
    print("✅ LLM 令牌流式传输完成")
    print()


# ==================== 3. 工具更新流式传输 ====================

def example_tool_updates_streaming():
    """
    示例 3: 工具更新流式传输
    
    使用 get_stream_writer 在工具执行时流式传输自定义更新。
    注意：如果您在工具内部添加 get_stream_writer，您将无法在 LangGraph 执行上下文之外调用该工具。
    """
    print("=" * 80)
    print("示例 3: 工具更新流式传输 (stream_mode='custom')")
    print("=" * 80)
    print()
    
    # 创建 QwenClient
    qwen_client = QwenClient(
        model="qwen-plus",
        temperature=0.7,
    )
    
    # 创建代理
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[get_weather],
    )
    
    # 流式传输工具更新
    print("开始流式传输工具更新...")
    print("-" * 80)
    
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "上海今天天气怎么样？"}]},
        stream_mode="custom"
    ):
        print(f"🔧 工具更新:")
        print(f"   {chunk}")
        print()
    
    print("✅ 工具更新流式传输完成")
    print()


# ==================== 4. 多种模式同时流式传输 ====================

def example_multiple_streaming_modes():
    """
    示例 4: 多种模式同时流式传输
    
    通过将流模式作为列表传递来指定多种流式传输模式：
    stream_mode=["updates", "messages", "custom"]
    """
    print("=" * 80)
    print("示例 4: 多种模式同时流式传输")
    print("stream_mode=['updates', 'messages', 'custom']")
    print("=" * 80)
    print()
    
    # 创建 QwenClient
    qwen_client = QwenClient(
        model="qwen-plus",
        temperature=0.7,
    )
    
    # 创建代理
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[get_weather],
    )
    
    # 同时流式传输多种模式
    print("开始同时流式传输多种模式...")
    print("-" * 80)
    
    for stream_mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": "深圳今天天气怎么样？"}]},
        stream_mode=["updates", "messages", "custom"]
    ):
        print(f"📊 流模式: {stream_mode}")
        print(f"   内容: {chunk}")
        print()
    
    print("✅ 多种模式流式传输完成")
    print()


# ==================== 5. 异步流式传输示例 ====================

async def example_async_streaming():
    """
    示例 5: 异步流式传输
    
    使用 astream() 方法进行异步流式传输。
    """
    print("=" * 80)
    print("示例 5: 异步流式传输 (astream)")
    print("=" * 80)
    print()
    
    # 创建 QwenClient
    qwen_client = QwenClient(
        model="qwen-plus",
        temperature=0.7,
    )
    
    # 创建代理
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[get_weather],
    )
    
    # 异步流式传输
    print("开始异步流式传输...")
    print("-" * 80)
    
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "广州今天天气怎么样？"}]},
        stream_mode="updates"
    ):
        print(f"📦 异步更新块:")
        print(f"   {chunk}")
        print()
    
    print("✅ 异步流式传输完成")
    print()


# ==================== 6. 禁用流式传输 ====================

def example_disable_streaming():
    """
    示例 6: 禁用流式传输
    
    在某些应用程序中，您可能需要禁用给定模型的单个令牌流式传输。
    这在多代理系统中很有用，用于控制哪些代理流式传输其输出。
    
    注意：具体的禁用方法取决于使用的 LLM 客户端实现。
    对于 QwenClient，可以通过配置模型参数来控制流式传输行为。
    """
    print("=" * 80)
    print("示例 6: 禁用流式传输")
    print("=" * 80)
    print()
    
    # 创建 QwenClient（可以配置流式传输相关参数）
    qwen_client = QwenClient(
        model="qwen-plus",
        temperature=0.7,
        # 注意：具体的禁用流式传输的方法取决于 QwenClient 的实现
        # 这里仅作为示例说明概念
    )
    
    # 创建代理
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[get_weather],
    )
    
    # 使用 invoke 而不是 stream 来禁用流式传输
    print("使用 invoke 方法（非流式）...")
    print("-" * 80)
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]}
    )
    
    print("完整响应:")
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"   {last_message.content}")
    
    print()
    print("✅ 非流式调用完成")
    print()
    print("💡 提示: 在多代理系统中，您可以为不同的代理配置不同的流式传输行为")
    print()


# ==================== 主函数 ====================

def main():
    """运行所有流式传输示例"""
    try:
        # 示例 1: 代理进度流式传输
        example_agent_progress_streaming()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 2: LLM 令牌流式传输
        example_llm_token_streaming()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 3: 工具更新流式传输
        example_tool_updates_streaming()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 4: 多种模式同时流式传输
        example_multiple_streaming_modes()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 6: 禁用流式传输
        example_disable_streaming()
        
        # 注意：示例 5（异步流式传输）需要异步环境运行
        # 如果需要运行异步示例，可以使用以下代码：
        # import asyncio
        # asyncio.run(example_async_streaming())
        
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

