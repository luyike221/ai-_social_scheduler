"""
模型配置示例

本示例展示了如何在 LangGraph 中配置代理使用的聊天模型，包括：

1. 按名称指定模型 - 使用模型名称字符串配置代理
2. 使用 init_chat_model - 简化模型初始化并提供可配置参数
3. 使用特定提供商的 LLM - 直接实例化特定提供商的模型类
4. 禁用流式传输 - 在初始化模型时禁用单个 LLM 令牌的流式传输
5. 添加模型回退 - 为不同的模型或不同的 LLM 提供商添加回退

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.a-base.model_config_example

"""

import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ai_social_scheduler.client import QwenClient


# ==================== 工具定义 ====================

@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式
    
    Args:
        expression: 数学表达式，如 "2 + 2" 或 "10 * 5"
    
    Returns:
        str: 计算结果
    """
    try:
        # 简单的安全计算（仅支持基本运算）
        result = eval(expression.replace(" ", ""))
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# ==================== 1. 按名称指定模型 ====================

def example_model_by_name():
    """
    示例 1: 按名称指定模型
    
    使用模型名称字符串配置代理。这是最简单的方式。
    
    注意：此方法需要模型提供商支持通过名称字符串初始化。
    对于 OpenAI、Anthropic 等，可以直接使用。
    对于自定义客户端（如 QwenClient），需要先创建客户端实例。
    """
    print("=" * 80)
    print("示例 1: 按名称指定模型")
    print("=" * 80)
    print()
    
    # 方式 1: 使用 QwenClient（项目自定义客户端）
    print("方式 1: 使用 QwenClient")
    print("-" * 80)
    
    qwen_client = QwenClient(
        model="qwen-plus",
        temperature=0.7,
    )
    
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[calculator],
    )
    
    response = agent.invoke({
        "messages": [HumanMessage(content="计算 15 * 8 的结果")]
    })
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"模型: {qwen_client.model}")
        print(f"问题: 计算 15 * 8 的结果")
        print(f"回答: {last_message.content}")
    
    print()
    
    # 方式 2: 使用 OpenAI（如果配置了 OPENAI_API_KEY）
    print("方式 2: 使用 OpenAI（需要配置 OPENAI_API_KEY）")
    print("-" * 80)
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            
            openai_model = ChatOpenAI(
                model="gpt-4",
                temperature=0,
            )
            
            agent = create_react_agent(
                model=openai_model,
                tools=[calculator],
            )
            
            response = agent.invoke({
                "messages": [HumanMessage(content="计算 20 + 30 的结果")]
            })
            
            if "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                print(f"模型: gpt-4")
                print(f"问题: 计算 20 + 30 的结果")
                print(f"回答: {last_message.content}")
        except Exception as e:
            print(f"⚠️  OpenAI 配置错误: {e}")
            print("提示: 请设置 OPENAI_API_KEY 环境变量")
    else:
        print("⚠️  未配置 OPENAI_API_KEY，跳过 OpenAI 示例")
    
    print()
    print("✅ 按名称指定模型示例完成")
    print()


# ==================== 2. 使用 init_chat_model ====================

def example_init_chat_model():
    """
    示例 2: 使用 init_chat_model
    
    init_chat_model 工具简化了模型初始化，并提供了可配置参数。
    支持多种模型提供商：OpenAI、Anthropic、Azure、Google Gemini、AWS Bedrock 等。
    
    注意：需要安装相应的依赖包，如：
    - pip install -U "langchain[openai]"
    - pip install -U "langchain[anthropic]"
    """
    print("=" * 80)
    print("示例 2: 使用 init_chat_model")
    print("=" * 80)
    print()
    
    # 尝试使用 init_chat_model（如果可用）
    try:
        from langchain.chat_models import init_chat_model
        
        print("使用 init_chat_model 初始化模型...")
        print("-" * 80)
        
        # 示例：使用 OpenAI（如果配置了 API Key）
        if os.getenv("OPENAI_API_KEY"):
            try:
                model = init_chat_model(
                    "openai:gpt-4",
                    temperature=0,
                    max_tokens=1000,
                )
                
                agent = create_react_agent(
                    model=model,
                    tools=[calculator],
                )
                
                response = agent.invoke({
                    "messages": [HumanMessage(content="计算 100 / 4 的结果")]
                })
                
                if "messages" in response and response["messages"]:
                    last_message = response["messages"][-1]
                    print(f"模型: openai:gpt-4")
                    print(f"问题: 计算 100 / 4 的结果")
                    print(f"回答: {last_message.content}")
            except Exception as e:
                print(f"⚠️  OpenAI 初始化错误: {e}")
        
        # 示例：使用 Anthropic（如果配置了 API Key）
        elif os.getenv("ANTHROPIC_API_KEY"):
            try:
                model = init_chat_model(
                    "anthropic:claude-3-5-sonnet-latest",
                    temperature=0,
                )
                
                agent = create_react_agent(
                    model=model,
                    tools=[calculator],
                )
                
                response = agent.invoke({
                    "messages": [HumanMessage(content="计算 50 - 25 的结果")]
                })
                
                if "messages" in response and response["messages"]:
                    last_message = response["messages"][-1]
                    print(f"模型: anthropic:claude-3-5-sonnet-latest")
                    print(f"问题: 计算 50 - 25 的结果")
                    print(f"回答: {last_message.content}")
            except Exception as e:
                print(f"⚠️  Anthropic 初始化错误: {e}")
        else:
            print("⚠️  未配置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY")
            print("提示: 请设置相应的 API Key 环境变量")
            print("或者使用 QwenClient（项目默认客户端）")
            
            # 使用 QwenClient 作为替代示例
            qwen_client = QwenClient(
                model="qwen-plus",
                temperature=0,
            )
            
            agent = create_react_agent(
                model=qwen_client.client,
                tools=[calculator],
            )
            
            response = agent.invoke({
                "messages": [HumanMessage(content="计算 12 * 6 的结果")]
            })
            
            if "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                print(f"模型: {qwen_client.model} (使用 QwenClient)")
                print(f"问题: 计算 12 * 6 的结果")
                print(f"回答: {last_message.content}")
        
    except ImportError:
        print("⚠️  init_chat_model 不可用")
        print("提示: 请安装 langchain 包: pip install -U langchain")
        print("或者使用特定提供商的模型类（见示例 3）")
    
    print()
    print("✅ 使用 init_chat_model 示例完成")
    print()


# ==================== 3. 使用特定提供商的 LLM ====================

def example_provider_specific_llm():
    """
    示例 3: 使用特定提供商的 LLM
    
    如果模型提供商无法通过 init_chat_model 获得，您可以直接实例化该提供商的模型类。
    该模型必须实现 BaseChatModel 接口并支持工具调用。
    
    本示例展示：
    - 使用 QwenClient（项目自定义客户端）
    - 使用 OpenAI ChatOpenAI
    - 使用 Anthropic ChatAnthropic
    """
    print("=" * 80)
    print("示例 3: 使用特定提供商的 LLM")
    print("=" * 80)
    print()
    
    # 方式 1: 使用 QwenClient（项目自定义客户端）
    print("方式 1: 使用 QwenClient")
    print("-" * 80)
    
    qwen_client = QwenClient(
        model="qwen-plus",
        temperature=0.7,
        max_tokens=2048,
    )
    
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[calculator],
    )
    
    response = agent.invoke({
        "messages": [HumanMessage(content="计算 8 * 9 的结果")]
    })
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"模型: {qwen_client.model}")
        print(f"温度: {qwen_client.temperature}")
        print(f"最大 tokens: {qwen_client.max_tokens}")
        print(f"问题: 计算 8 * 9 的结果")
        print(f"回答: {last_message.content}")
    
    print()
    
    # 方式 2: 使用 OpenAI ChatOpenAI
    print("方式 2: 使用 OpenAI ChatOpenAI")
    print("-" * 80)
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            
            model = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                max_tokens=2048,
            )
            
            agent = create_react_agent(
                model=model,
                tools=[calculator],
            )
            
            response = agent.invoke({
                "messages": [HumanMessage(content="计算 7 * 7 的结果")]
            })
            
            if "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                print(f"模型: gpt-4")
                print(f"温度: 0")
                print(f"最大 tokens: 2048")
                print(f"问题: 计算 7 * 7 的结果")
                print(f"回答: {last_message.content}")
        except Exception as e:
            print(f"⚠️  OpenAI 错误: {e}")
    else:
        print("⚠️  未配置 OPENAI_API_KEY，跳过 OpenAI 示例")
    
    print()
    
    # 方式 3: 使用 Anthropic ChatAnthropic
    print("方式 3: 使用 Anthropic ChatAnthropic")
    print("-" * 80)
    
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            
            model = ChatAnthropic(
                model="claude-3-5-sonnet-latest",
                temperature=0,
                max_tokens=2048,
            )
            
            agent = create_react_agent(
                model=model,
                tools=[calculator],
            )
            
            response = agent.invoke({
                "messages": [HumanMessage(content="计算 6 * 6 的结果")]
            })
            
            if "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                print(f"模型: claude-3-5-sonnet-latest")
                print(f"温度: 0")
                print(f"最大 tokens: 2048")
                print(f"问题: 计算 6 * 6 的结果")
                print(f"回答: {last_message.content}")
        except Exception as e:
            print(f"⚠️  Anthropic 错误: {e}")
    else:
        print("⚠️  未配置 ANTHROPIC_API_KEY，跳过 Anthropic 示例")
    
    print()
    print("✅ 使用特定提供商的 LLM 示例完成")
    print()


# ==================== 4. 禁用流式传输 ====================

def example_disable_streaming():
    """
    示例 4: 禁用流式传输
    
    要禁用单个 LLM 令牌的流式传输，可以在初始化模型时设置 disable_streaming=True。
    这在多代理系统中很有用，用于控制哪些代理流式传输其输出。
    
    注意：具体的禁用方法取决于使用的 LLM 客户端实现。
    """
    print("=" * 80)
    print("示例 4: 禁用流式传输")
    print("=" * 80)
    print()
    
    # 方式 1: 使用 init_chat_model 禁用流式传输
    print("方式 1: 使用 init_chat_model 禁用流式传输")
    print("-" * 80)
    
    try:
        from langchain.chat_models import init_chat_model
        
        if os.getenv("OPENAI_API_KEY"):
            try:
                model = init_chat_model(
                    "openai:gpt-4",
                    temperature=0,
                    disable_streaming=True,  # 禁用流式传输
                )
                
                agent = create_react_agent(
                    model=model,
                    tools=[calculator],
                )
                
                print("使用非流式模式调用代理...")
                response = agent.invoke({
                    "messages": [HumanMessage(content="计算 5 * 5 的结果")]
                })
                
                if "messages" in response and response["messages"]:
                    last_message = response["messages"][-1]
                    print(f"模型: openai:gpt-4 (disable_streaming=True)")
                    print(f"回答: {last_message.content}")
            except Exception as e:
                print(f"⚠️  错误: {e}")
        else:
            print("⚠️  未配置 OPENAI_API_KEY")
    except ImportError:
        print("⚠️  init_chat_model 不可用")
    
    print()
    
    # 方式 2: 使用 QwenClient（通过 invoke 而不是 stream）
    print("方式 2: 使用 QwenClient（非流式调用）")
    print("-" * 80)
    
    qwen_client = QwenClient(
        model="qwen-plus",
        temperature=0.7,
    )
    
    agent = create_react_agent(
        model=qwen_client.client,
        tools=[calculator],
    )
    
    print("使用 invoke 方法（非流式）...")
    response = agent.invoke({
        "messages": [HumanMessage(content="计算 4 * 4 的结果")]
    })
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"模型: {qwen_client.model}")
        print(f"调用方式: invoke (非流式)")
        print(f"回答: {last_message.content}")
    
    print()
    print("💡 提示: 在多代理系统中，您可以为不同的代理配置不同的流式传输行为")
    print("✅ 禁用流式传输示例完成")
    print()


# ==================== 5. 添加模型回退 ====================

def example_model_fallbacks():
    """
    示例 5: 添加模型回退
    
    您可以使用 model.with_fallbacks([...]) 为不同的模型或不同的 LLM 提供商添加回退。
    当主模型失败时，系统会自动尝试使用回退模型。
    
    这对于提高系统的可靠性和可用性非常有用。
    """
    print("=" * 80)
    print("示例 5: 添加模型回退")
    print("=" * 80)
    print()
    
    # 方式 1: 使用 init_chat_model 添加回退
    print("方式 1: 使用 init_chat_model 添加回退")
    print("-" * 80)
    
    try:
        from langchain.chat_models import init_chat_model
        
        # 创建带回退的模型
        # 主模型：Anthropic Claude
        # 回退模型：OpenAI GPT-4
        if os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENAI_API_KEY"):
            try:
                model_with_fallbacks = (
                    init_chat_model("anthropic:claude-3-5-sonnet-latest")
                    .with_fallbacks([
                        init_chat_model("openai:gpt-4"),
                    ])
                )
                
                agent = create_react_agent(
                    model=model_with_fallbacks,
                    tools=[calculator],
                )
                
                print("使用带回退的模型调用代理...")
                print("主模型: anthropic:claude-3-5-sonnet-latest")
                print("回退模型: openai:gpt-4")
                print("-" * 80)
                
                response = agent.invoke({
                    "messages": [HumanMessage(content="计算 3 * 3 的结果")]
                })
                
                if "messages" in response and response["messages"]:
                    last_message = response["messages"][-1]
                    print(f"回答: {last_message.content}")
            except Exception as e:
                print(f"⚠️  错误: {e}")
        else:
            print("⚠️  需要配置 ANTHROPIC_API_KEY 和 OPENAI_API_KEY")
            print("提示: 使用 QwenClient 作为替代示例")
            
            # 使用 QwenClient 作为替代示例
            qwen_client = QwenClient(
                model="qwen-plus",
                temperature=0.7,
            )
            
            agent = create_react_agent(
                model=qwen_client.client,
                tools=[calculator],
            )
            
            response = agent.invoke({
                "messages": [HumanMessage(content="计算 2 * 2 的结果")]
            })
            
            if "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                print(f"模型: {qwen_client.model} (单模型，无回退)")
                print(f"回答: {last_message.content}")
    
    except ImportError:
        print("⚠️  init_chat_model 不可用")
        print("提示: 请安装 langchain 包")
    
    print()
    
    # 方式 2: 使用多个 QwenClient 模型作为回退
    print("方式 2: 使用多个 QwenClient 模型作为回退")
    print("-" * 80)
    
    try:
        from langchain.chat_models import init_chat_model
        
        # 注意：QwenClient 可能不支持 with_fallbacks
        # 这里展示概念，实际使用时需要根据具体实现调整
        qwen_client_primary = QwenClient(
            model="qwen-plus",
            temperature=0.7,
        )
        
        qwen_client_fallback = QwenClient(
            model="qwen-turbo",  # 使用更快的模型作为回退
            temperature=0.7,
        )
        
        # 如果支持 with_fallbacks，可以这样使用：
        # model_with_fallbacks = qwen_client_primary.client.with_fallbacks([
        #     qwen_client_fallback.client
        # ])
        
        # 这里直接使用主模型
        agent = create_react_agent(
            model=qwen_client_primary.client,
            tools=[calculator],
        )
        
        print("主模型: qwen-plus")
        print("回退模型: qwen-turbo (概念示例)")
        print("-" * 80)
        
        response = agent.invoke({
            "messages": [HumanMessage(content="计算 1 * 1 的结果")]
        })
        
        if "messages" in response and response["messages"]:
            last_message = response["messages"][-1]
            print(f"回答: {last_message.content}")
    
    except Exception as e:
        print(f"⚠️  错误: {e}")
    
    print()
    print("💡 提示: 模型回退可以提高系统的可靠性和可用性")
    print("✅ 添加模型回退示例完成")
    print()


# ==================== 主函数 ====================

def main():
    """运行所有模型配置示例"""
    try:
        # 示例 1: 按名称指定模型
        example_model_by_name()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 2: 使用 init_chat_model
        example_init_chat_model()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 3: 使用特定提供商的 LLM
        example_provider_specific_llm()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 4: 禁用流式传输
        example_disable_streaming()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 5: 添加模型回退
        example_model_fallbacks()
        
    except ValueError as e:
        print(f"\n❌ 配置错误: {e}")
        print("请确保在 .env 文件中配置了相应的 API Key")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

