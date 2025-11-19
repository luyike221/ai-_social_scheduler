"""
使用多个 MCP 服务器创建 LangGraph Agent（使用 Qwen 模型）

本示例展示如何：
1. 连接多个 MCP 服务器（小红书 + 高德地图）
2. 获取所有服务器的工具
3. 合并工具列表
4. 使用 Qwen 模型创建 LangGraph Agent
5. 使用 Agent 处理跨服务的请求

安装依赖：
pip install langchain-mcp-adapters

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.g-mcp.multiple_mcp_servers_example

"""

import asyncio

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from ai_social_scheduler.client import QwenClient
from ai_social_scheduler.config.model_config import model_config


async def example_multiple_mcp_servers():
    """
    示例：连接多个 MCP 服务器并使用 LangGraph Agent
    """
    print("=" * 80)
    print("示例：使用多个 MCP 服务器创建 LangGraph Agent（Qwen 模型）")
    print("=" * 80)
    print()
    
    try:
        # 从配置中获取 API Key（用于高德地图服务）
        alibaba_config = model_config.get_alibaba_bailian_config()
        api_key = alibaba_config.api_key
        
        # 连接多个 MCP 服务器
        print("📡 正在连接多个 MCP 服务器...")
        print()
        
        client = MultiServerMCPClient({
            # MCP 服务器 1: 小红书服务
            "xiaohongshu": {
                "url": "http://127.0.0.1:8002/mcp",
                "transport": "streamable_http",
            },
            # MCP 服务器 2: 高德地图服务
            "amap-maps": {
                "url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
                "transport": "streamable_http",
                "headers": {
                    "Authorization": f"Bearer {api_key}"
                }
            }
        })
        
        # 获取所有服务器的工具（自动合并）
        print("🔧 获取所有 MCP 服务器的工具...")
        tools = await client.get_tools()
        print(f"✅ 总共获取到 {len(tools)} 个工具")
        print()
        
        # 按服务器分组显示工具
        print("📋 工具列表（按服务器分组）:")
        xiaohongshu_tools = [t for t in tools if t.name.startswith("xiaohongshu_")]
        amap_tools = [t for t in tools if t.name.startswith("amap_") or "地图" in t.description or "地理" in t.description]
        other_tools = [t for t in tools if t not in xiaohongshu_tools and t not in amap_tools]
        
        if xiaohongshu_tools:
            print(f"\n   📱 小红书服务工具 ({len(xiaohongshu_tools)} 个):")
            for tool in xiaohongshu_tools:
                print(f"      - {tool.name}")
        
        if amap_tools:
            print(f"\n   🗺️  高德地图服务工具 ({len(amap_tools)} 个):")
            for tool in amap_tools:
                print(f"      - {tool.name}")
        
        if other_tools:
            print(f"\n   🔧 其他工具 ({len(other_tools)} 个):")
            for tool in other_tools:
                print(f"      - {tool.name}")
        
        print()
        
        # 创建 Qwen 模型客户端
        print("🤖 创建 Qwen 模型客户端...")
        qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
        print(f"✅ 模型: {qwen_client.model}, 温度: {qwen_client.temperature}")
        print()
        
        # 创建 LangGraph Agent（使用所有工具）
        print("🔧 创建 LangGraph Agent（使用所有 MCP 工具）...")
        agent = create_react_agent(
            qwen_client.client,
            tools  # 所有工具自动合并
        )
        print("✅ Agent 创建成功")
        print()
        
        # 示例 1: 使用小红书工具
        print("=" * 80)
        print("示例 1: 使用小红书工具")
        print("=" * 80)
        print("💬 请求: 检查一下我的小红书登录状态")
        print("-" * 80)
        
        response1 = await agent.ainvoke({
            "messages": [HumanMessage(content="检查一下我的小红书登录状态")]
        })
        
        if "messages" in response1 and response1["messages"]:
            last_message = response1["messages"][-1]
            print(f"📝 Agent 回复:")
            print(f"   {last_message.content}")
        print()
        
        # 示例 2: 使用高德地图工具（如果可用）
        if amap_tools:
            print("=" * 80)
            print("示例 2: 使用高德地图工具")
            print("=" * 80)
            print("💬 请求: 查询北京的天气")
            print("-" * 80)
            
            response2 = await agent.ainvoke({
                "messages": [HumanMessage(content="查询北京的天气")]
            })
            
            if "messages" in response2 and response2["messages"]:
                last_message = response2["messages"][-1]
                print(f"📝 Agent 回复:")
                print(f"   {last_message.content}")
            print()
        
        # 示例 3: 跨服务使用（结合多个服务）
        print("=" * 80)
        print("示例 3: 跨服务使用")
        print("=" * 80)
        print("💬 请求: 先检查我的小红书登录状态，然后查询上海的天气")
        print("-" * 80)
        
        response3 = await agent.ainvoke({
            "messages": [HumanMessage(content="先检查我的小红书登录状态，然后查询上海的天气")]
        })
        
        if "messages" in response3 and response3["messages"]:
            last_message = response3["messages"][-1]
            print(f"📝 Agent 回复:")
            print(f"   {last_message.content}")
        print()
        
        print("=" * 80)
        print("✅ 多个 MCP 服务器示例完成")
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


async def example_custom_multiple_servers():
    """
    示例：自定义多个 MCP 服务器配置
    """
    print("=" * 80)
    print("示例：自定义多个 MCP 服务器配置")
    print("=" * 80)
    print()
    
    print("💡 配置多个 MCP 服务器的方法:")
    print("""
    client = MultiServerMCPClient({
        # 服务器 1: HTTP 传输
        "server1": {
            "url": "http://127.0.0.1:8002/mcp",
            "transport": "streamable_http",
        },
        # 服务器 2: HTTP 传输（带认证）
        "server2": {
            "url": "https://api.example.com/mcp",
            "transport": "streamable_http",
            "headers": {
                "Authorization": "Bearer your-api-key"
            }
        },
        # 服务器 3: stdio 传输（本地进程）
        "server3": {
            "command": "python",
            "args": ["-m", "your_mcp_server"],
            "transport": "stdio",
        }
    })
    
    # 获取所有服务器的工具（自动合并）
    tools = await client.get_tools()
    
    # 创建 Agent（使用所有工具）
    agent = create_react_agent(
        model=your_model,
        tools=tools
    )
    """)
    print()
    print("✅ 配置示例完成")
    print()


async def main():
    """主函数"""
    try:
        # 示例 1: 使用多个 MCP 服务器
        await example_multiple_mcp_servers()
        
        print("\n" + "=" * 80 + "\n")
        
        # 示例 2: 配置说明
        await example_custom_multiple_servers()
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

