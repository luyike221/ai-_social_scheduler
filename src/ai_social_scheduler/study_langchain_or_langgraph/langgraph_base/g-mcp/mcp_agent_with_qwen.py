"""
使用 MCP 工具创建 LangGraph Agent（使用 Qwen 模型）

本示例展示如何：
1. 连接小红书 MCP 服务
2. 获取 MCP 工具
3. 使用 Qwen 模型创建 LangGraph Agent
4. 使用 Agent 处理请求

安装依赖：
pip install langchain-mcp-adapters

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.g-mcp.mcp_agent_with_qwen

"""

import asyncio

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from ai_social_scheduler.client import QwenClient


async def example_mcp_agent_with_qwen():
    """
    示例：将 MCP 工具与 LangGraph Agent 结合使用（使用 Qwen 模型）
    """
    print("=" * 80)
    print("示例：使用 MCP 工具创建 LangGraph Agent（Qwen 模型）")
    print("=" * 80)
    print()
    
    try:
        # 连接 MCP 服务
        client = MultiServerMCPClient({
            "xiaohongshu": {
                "url": "http://127.0.0.1:8002/mcp",
                "transport": "streamable_http",
            }
        })
        
        # 获取工具（新 API：直接调用 get_tools()）
        print("📡 正在连接 MCP 服务...")
        tools = await client.get_tools()
        print(f"✅ 获取到 {len(tools)} 个工具")
        print()
        
        # 显示可用工具
        print("📋 可用工具列表:")
        for tool in tools:
            print(f"   - {tool.name}")
        print()
        
        # 创建 Qwen 模型客户端
        print("🤖 创建 Qwen 模型客户端...")
        qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
        print(f"✅ 模型: {qwen_client.model}, 温度: {qwen_client.temperature}")
        print()
        
        # 创建 LangGraph Agent
        print("🔧 创建 LangGraph Agent...")
        agent = create_react_agent(
            qwen_client.client,
            tools
        )
        print("✅ Agent 创建成功")
        print()
        
        # 使用 Agent 处理请求
        print("💬 使用 Agent 处理请求...")
        print("   请求: 检查一下我的小红书登录状态")
        print("-" * 80)
        
        response = await agent.ainvoke({
            "messages": [HumanMessage(content="检查一下我的小红书登录状态")]
        })
        
        if "messages" in response and response["messages"]:
            last_message = response["messages"][-1]
            print(f"📝 Agent 回复:")
            print(f"   {last_message.content}")
        else:
            print(f"📝 Agent 响应: {response}")
        
        print()
        print("=" * 80)
        print("✅ Agent 示例完成")
        print("=" * 80)
            
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请安装: pip install langchain-mcp-adapters")
        import traceback
        traceback.print_exc()
        return
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return


async def main():
    """主函数"""
    try:
        await example_mcp_agent_with_qwen()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

