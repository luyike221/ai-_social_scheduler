"""
LangGraph 快速入门示例
使用 QwenClient（通义千问）作为大模型

本示例展示了如何使用 LangGraph 创建代理系统，包括：
1. 创建代理
2. 配置 LLM
3. 添加自定义提示
4. 添加记忆
5. 配置结构化输出

uv run python examples/langgraph_quickstart_example.py
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from ...client import QwenClient
"""
执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.base.langgraph_quickstart

"""

# ==================== 工具定义 ====================

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 这是一个示例工具，实际应用中应该调用真实的天气 API
    return f"{city} 的天气：晴朗，温度 25°C"


# ==================== 综合示例：LangGraph 完整功能演示 ====================

def comprehensive_example():
    """综合示例：展示 LangGraph 的核心功能"""
    print("=" * 60)
    print("LangGraph 综合示例 - 使用 QwenClient（通义千问）")
    print("=" * 60)
    print()

    # 创建 QwenClient
    qwen_client = QwenClient(
        model="qwen-plus",
        temperature=0.7,
    )

    # ========== 步骤 1: 添加记忆功能和结构化输出 ==========
    print("【步骤 1】添加记忆功能和结构化输出")
    print("-" * 60)
    
    # 创建检查点保存器（用于记忆）
    checkpointer = InMemorySaver()
    
    # 定义响应结构（使用 Pydantic 模型）
    # 注意：LangGraph 的 response_format 需要模型支持 .with_structured_output
    class WeatherResponse(BaseModel):
        """天气响应结构"""
        city: str = Field(description="城市名称")
        conditions: str = Field(default="", description="天气状况，如：晴朗、多云、雨天等")
        temperature: str = Field(default="", description="温度，如：25°C")
        description: str = Field(default="", description="天气的详细描述")
    
    # 创建同时支持记忆和结构化输出的代理
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[get_weather],
        checkpointer=checkpointer,  # 启用记忆
        response_format=WeatherResponse,  # 指定响应格式
        prompt="你是一个专业的天气助手。",
    )


    
    print("✅ 已创建代理（支持记忆和结构化输出）")
    
    # 配置：使用 thread_id 来标识对话会话
    config = {"configurable": {"thread_id": "comprehensive-example-1"}}
    
    # 第一轮对话
    print("\n第一轮对话:")
    response1 = agent.invoke(
        {"messages": [HumanMessage(content="请用 JSON 格式告诉我北京今天天气怎么样？")]},
        config,
    )
    print(f"用户: 请用 JSON 格式告诉我北京今天天气怎么样？")
    print(f"助手: {response1['messages'][-1].content}")
    
    # 获取结构化响应
    if "structured_response" in response1:
        structured = response1["structured_response"]
        print(f"\n📊 结构化响应:")
        print(f"  城市: {structured.city}")
        print(f"  天气状况: {structured.conditions}")
        print(f"  温度: {structured.temperature}")
        print(f"  描述: {structured.description}")
    
    # 第二轮对话（会自动包含之前的对话历史）
    print("\n第二轮对话（带上下文记忆）:")
    response2 = agent.invoke(
        {"messages": [HumanMessage(content="请用 JSON 格式告诉我那上海呢？")]},
        config,  # 使用相同的 thread_id，会自动加载之前的对话历史
    )
    print(f"用户: 请用 JSON 格式告诉我那上海呢？")
    print(f"助手: {response2['messages'][-1].content}")
    
    # 获取结构化响应
    if "structured_response" in response2:
        structured = response2["structured_response"]
        print(f"\n📊 结构化响应:")
        print(f"  城市: {structured.city}")
        print(f"  天气状况: {structured.conditions}")
        print(f"  温度: {structured.temperature}")
        print(f"  描述: {structured.description}")
    print()


# ==================== 主函数 ====================

def main():
    """运行综合示例"""
    try:
        comprehensive_example()
    except ValueError as e:
        print(f"\n❌ 配置错误: {e}")
        print("请确保在 .env 文件中配置了 ALIBABA_BAILIAN_API_KEY")
        import traceback
        traceback.print_exc()
    except TypeError as e:
        if "prompt" in str(e) and "list" in str(e):
            print(f"\n❌ 类型错误: {e}")
            print("提示: prompt 参数必须是字符串类型，不能是列表")
            print("请检查代码中所有 create_react_agent 的 prompt 参数")
        else:
            print(f"\n❌ 类型错误: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

