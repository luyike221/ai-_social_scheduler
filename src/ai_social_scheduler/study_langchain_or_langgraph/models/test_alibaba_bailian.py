"""测试使用 LangChain OpenAI 接口连接阿里百炼"""

import asyncio
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ...config import model_config


async def test_alibaba_bailian_with_langchain():
    """测试使用 LangChain OpenAI 接口连接阿里百炼"""
    
    try:
        # 获取阿里百炼配置
        bailian_config = model_config.get_alibaba_bailian_config()
        
        print("=" * 60)
        print("测试 LangChain OpenAI 接口连接阿里百炼")
        print("=" * 60)
        print(f"Endpoint: {bailian_config.endpoint}")
        print(f"Model: {bailian_config.model}")
        print(f"Temperature: {bailian_config.temperature}")
        print("-" * 60)
        
        # 创建 LangChain ChatOpenAI 客户端
        # 阿里百炼提供了 OpenAI 兼容模式的 API
        # 注意：阿里百炼的兼容模式使用 API Key 作为认证，格式为 "sk-{api_key}"
        # 如果配置中有 API Secret，可能需要组合使用
        
        # 构建认证 key（阿里百炼兼容模式通常只需要 API Key）
        # 如果 API Key 不是以 "sk-" 开头，可能需要添加前缀
        api_key = bailian_config.api_key
        if not api_key.startswith("sk-"):
            api_key = f"sk-{api_key}"
        
        llm = ChatOpenAI(
            model=bailian_config.model,
            openai_api_key=api_key,
            base_url=bailian_config.endpoint,
            temperature=bailian_config.temperature,
            timeout=bailian_config.timeout,
            max_tokens=bailian_config.max_tokens,
        )
        
        # 准备测试消息
        messages = [
            SystemMessage(content="你是一个有用的AI助手。"),
            HumanMessage(content="你好，请简单介绍一下你自己。"),
        ]
        
        print("\n发送测试消息...")
        print(f"用户消息: {messages[-1].content}")
        print("-" * 60)
        
        # 调用模型
        response = await llm.ainvoke(messages)
        
        print("\n✅ 连接成功！")
        print(f"模型回复: {response.content}")
        print("=" * 60)
        
        return True
        
    except ValueError as e:
        print(f"\n❌ 配置错误: {e}")
        print("请确保在 .env 文件中配置了以下变量：")
        print("  - ALIBABA_BAILIAN_API_KEY")
        print("  - ALIBABA_BAILIAN_API_SECRET")
        return False
        
    except Exception as e:
        print(f"\n❌ 连接失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        print("\n可能的原因：")
        print("1. API Key 或 API Secret 配置错误")
        print("2. 网络连接问题")
        print("3. 阿里百炼的认证方式与 OpenAI 不完全兼容")
        print("4. 需要额外的认证头或参数")
        return False


async def test_streaming():
    """测试流式响应"""
    
    try:
        bailian_config = model_config.get_alibaba_bailian_config()
        
        print("\n" + "=" * 60)
        print("测试流式响应")
        print("=" * 60)
        
        # 构建认证 key
        api_key = bailian_config.api_key
        if not api_key.startswith("sk-"):
            api_key = f"sk-{api_key}"
        
        llm = ChatOpenAI(
            model=bailian_config.model,
            openai_api_key=api_key,
            base_url=bailian_config.endpoint,
            temperature=bailian_config.temperature,
            timeout=bailian_config.timeout,
        )
        
        messages = [
            HumanMessage(content="请用一句话介绍人工智能。"),
        ]
        
        print("流式响应: ", end="", flush=True)
        
        async for chunk in llm.astream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        
        print("\n✅ 流式响应测试完成")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 流式响应测试失败: {e}")
        return False


async def main():
    """主函数"""
    print("\n开始测试阿里百炼与 LangChain 的连接...\n")
    
    # 测试基本连接
    success = await test_alibaba_bailian_with_langchain()
    
    if success:
        # 如果基本连接成功，测试流式响应
        await test_streaming()
    
    print("\n测试完成！")


if __name__ == "__main__":
    asyncio.run(main())

