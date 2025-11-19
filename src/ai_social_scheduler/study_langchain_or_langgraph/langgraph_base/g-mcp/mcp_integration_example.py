"""
MCP 集成示例

模型上下文协议 (MCP) 是一个开放协议，用于标准化应用程序如何向语言模型提供工具和上下文。
LangGraph 代理可以通过 langchain-mcp-adapters 库使用 MCP 服务器上定义的工具。

本示例展示了：
1. 安装 langchain-mcp-adapters 库
2. 使用 MultiServerMCPClient 连接 MCP 服务器
3. 获取 MCP 工具并创建代理
4. 使用代理调用 MCP 工具

安装依赖：
pip install langchain-mcp-adapters

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.g-mcp.mcp_integration_example

"""

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from ai_social_scheduler.client import QwenClient


# ==================== 示例 1: 使用单个 MCP 服务器 ====================

async def example_single_mcp_server():
    """示例 1: 使用单个 MCP 服务器"""
    print("=" * 80)
    print("示例 1: 使用单个 MCP 服务器")
    print("=" * 80)
    print()
    
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        # 配置单个 MCP 服务器
        # 注意：需要实际的 MCP 服务器运行
        # 这里展示配置方式，实际使用时需要替换为真实的服务器路径或 URL
        
        print("💡 MCP 服务器配置示例:")
        print("""
        # 方式 1: 使用 stdio 传输（本地 Python 脚本）
        client = MultiServerMCPClient({
            "math": {
                "command": "python",
                "args": ["/path/to/math_server.py"],
                "transport": "stdio",
            }
        })
        
        # 方式 2: 使用 HTTP 传输（远程服务器）
        client = MultiServerMCPClient({
            "weather": {
                "url": "https://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        })
        """)
        
        print("⚠️  注意: 此示例需要实际的 MCP 服务器运行")
        print("   请参考 MCP 文档创建和运行 MCP 服务器")
        print()
        
        # 概念示例（需要实际的 MCP 服务器）
        # client = MultiServerMCPClient({
        #     "math": {
        #         "command": "python",
        #         "args": ["/path/to/math_server.py"],
        #         "transport": "stdio",
        #     }
        # })
        # 
        # tools = await client.get_tools()
        # 
        # qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
        # agent = create_react_agent(
        #     qwen_client.client,
        #     tools
        # )
        # 
        # response = await agent.ainvoke({
        #     "messages": [HumanMessage(content="计算 (3 + 5) × 12 的结果")]
        # })
        # 
        # if "messages" in response and response["messages"]:
        #     last_message = response["messages"][-1]
        #     print(f"问题: 计算 (3 + 5) × 12 的结果")
        #     print(f"回答: {last_message.content}")
        
    except ImportError:
        print("❌ 未安装 langchain-mcp-adapters")
        print("请运行: pip install langchain-mcp-adapters")
    
    print()
    print("✅ 单个 MCP 服务器示例完成")
    print()


# ==================== 示例 2: 使用多个 MCP 服务器 ====================

async def example_multiple_mcp_servers():
    """示例 2: 使用多个 MCP 服务器"""
    print("=" * 80)
    print("示例 2: 使用多个 MCP 服务器")
    print("=" * 80)
    print()
    
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        print("💡 多个 MCP 服务器配置示例:")
        print("""
        client = MultiServerMCPClient({
            # 数学服务器（stdio 传输）
            "math": {
                "command": "python",
                "args": ["/path/to/math_server.py"],
                "transport": "stdio",
            },
            # 天气服务器（HTTP 传输）
            "weather": {
                "url": "https://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        })
        
        # 获取所有服务器的工具
        tools = await client.get_tools()
        
        # 创建代理
        agent = create_react_agent(
            model=qwen_client.client,
            tools=tools
        )
        """)
        
        print("⚠️  注意: 此示例需要实际的 MCP 服务器运行")
        print("   配置示例展示了如何同时使用多个 MCP 服务器")
        print()
        
        # 概念示例（需要实际的 MCP 服务器）
        # client = MultiServerMCPClient({
        #     "math": {
        #         "command": "python",
        #         "args": ["/path/to/math_server.py"],
        #         "transport": "stdio",
        #     },
        #     "weather": {
        #         "url": "https://localhost:8000/mcp",
        #         "transport": "streamable_http",
        #     }
        # })
        # 
        # tools = await client.get_tools()
        # 
        # qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
        # agent = create_react_agent(
        #     qwen_client.client,
        #     tools
        # )
        # 
        # # 使用数学工具
        # math_response = await agent.ainvoke({
        #     "messages": [HumanMessage(content="计算 (3 + 5) × 12 的结果")]
        # })
        # 
        # # 使用天气工具
        # weather_response = await agent.ainvoke({
        #     "messages": [HumanMessage(content="纽约的天气怎么样？")]
        # })
        
    except ImportError:
        print("❌ 未安装 langchain-mcp-adapters")
        print("请运行: pip install langchain-mcp-adapters")
    
    print()
    print("✅ 多个 MCP 服务器示例完成")
    print()


# ==================== 示例 3: 完整的 MCP 使用流程 ====================

async def example_complete_workflow():
    """示例 3: 完整的 MCP 使用流程"""
    print("=" * 80)
    print("示例 3: 完整的 MCP 使用流程")
    print("=" * 80)
    print()
    
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        print("📋 完整使用流程:")
        print()
        print("1. 安装依赖:")
        print("   pip install langchain-mcp-adapters")
        print()
        print("2. 创建或启动 MCP 服务器")
        print("   - 使用 stdio 传输: 本地 Python 脚本")
        print("   - 使用 HTTP 传输: 远程 HTTP 服务器")
        print()
        print("3. 配置 MultiServerMCPClient:")
        print("""
        client = MultiServerMCPClient({
            "server_name": {
                "command": "python",  # 或 "url": "https://..."
                "args": ["/path/to/server.py"],  # 或 "transport": "streamable_http"
                "transport": "stdio",  # 或 "streamable_http"
            }
        })
        """)
        print()
        print("4. 获取工具并创建代理:")
        print("""
        tools = await client.get_tools()
        agent = create_react_agent(
            model=model,
            tools=tools
        )
        """)
        print()
        print("5. 使用代理调用 MCP 工具:")
        print("""
        response = await agent.ainvoke({
            "messages": [HumanMessage(content="你的问题")]
        })
        """)
        print()
        
        # 展示实际代码结构
        print("💻 完整代码示例:")
        print("""
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import HumanMessage
        
        # 1. 创建 MCP 客户端
        client = MultiServerMCPClient({
            "math": {
                "command": "python",
                "args": ["/path/to/math_server.py"],
                "transport": "stdio",
            }
        })
        
        # 2. 获取工具
        tools = await client.get_tools()
        
        # 3. 创建代理
        agent = create_react_agent(
            "anthropic:claude-3-7-sonnet-latest",
            tools
        )
        
        # 4. 使用代理
        response = await agent.ainvoke({
            "messages": [HumanMessage(content="计算 (3 + 5) × 12 的结果")]
        })
        
        print(response["messages"][-1].content)
        """)
        
    except ImportError:
        print("❌ 未安装 langchain-mcp-adapters")
        print("请运行: pip install langchain-mcp-adapters")
    
    print()
    print("✅ 完整流程示例完成")
    print()


# ==================== 示例 4: MCP 服务器类型 ====================

def example_mcp_server_types():
    """示例 4: MCP 服务器类型和传输方式"""
    print("=" * 80)
    print("示例 4: MCP 服务器类型和传输方式")
    print("=" * 80)
    print()
    
    print("📡 MCP 支持的传输方式:")
    print()
    print("1. stdio (标准输入输出)")
    print("   - 用于本地进程通信")
    print("   - 适合 Python 脚本、命令行工具")
    print("   - 配置示例:")
    print("""
    {
        "server_name": {
            "command": "python",
            "args": ["/path/to/server.py"],
            "transport": "stdio",
        }
    }
    """)
    print()
    print("2. streamable_http (HTTP 流式传输)")
    print("   - 用于远程 HTTP 服务器")
    print("   - 适合微服务、API 服务")
    print("   - 配置示例:")
    print("""
    {
        "server_name": {
            "url": "https://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
    """)
    print()
    print("💡 提示:")
    print("   - 可以同时使用多个不同类型的 MCP 服务器")
    print("   - 每个服务器可以提供多个工具")
    print("   - 工具会自动合并到代理的工具列表中")
    print()
    print("✅ MCP 服务器类型示例完成")
    print()


# ==================== 主函数 ====================

async def main():
    """运行所有 MCP 集成示例"""
    try:
        # 示例 1: 使用单个 MCP 服务器
        await example_single_mcp_server()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 2: 使用多个 MCP 服务器
        await example_multiple_mcp_servers()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 3: 完整的 MCP 使用流程
        await example_complete_workflow()
        print("\n" + "=" * 80 + "\n")
        
        # 示例 4: MCP 服务器类型
        example_mcp_server_types()
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

