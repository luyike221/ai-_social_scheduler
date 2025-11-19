#!/usr/bin/env python3
"""
使用 langchain_mcp_adapters 连接小红书 MCP 服务示例

前提条件：
1. 确保已安装依赖：pip install langchain-mcp-adapters
2. 确保 MCP 服务已启动（默认运行在 http://127.0.0.1:8002）

uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.g-mcp.connect_with_langchain
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


async def example_connect_xiaohongshu_mcp():
    """
    示例：连接小红书 MCP 服务（HTTP 传输方式）
    
    服务地址：http://127.0.0.1:8002/mcp
    如果服务运行在其他地址，请修改 URL
    """
    print("=" * 80)
    print("示例：使用 langchain_mcp_adapters 连接小红书 MCP 服务")
    print("=" * 80)
    print()
    
    try:
        # 方式 1: 使用 HTTP 传输（推荐，适用于已启动的服务）
        print("💡 方式 1: HTTP 传输（服务已启动）")
        print("   服务地址: http://127.0.0.1:8002/mcp")
        print()
        
        client = MultiServerMCPClient({
            "xiaohongshu": {
                "url": "http://127.0.0.1:8002/mcp",
                "transport": "streamable_http",
            }
        })
        
        # 连接并获取工具（新 API：直接调用 get_tools()）
        print("📡 正在连接小红书 MCP 服务...")
        tools = await client.get_tools()
        print("✅ 成功连接到小红书 MCP 服务")
        print()
        
        # 列出所有可用工具
        print(f"📋 可用工具数量: {len(tools)}")
        print(f"📋 工具列表:")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")
        print()
        
        # 示例：检查登录状态
        print("🔍 示例：检查登录状态")
        try:
            # 直接调用工具
            login_status_tool = next((t for t in tools if t.name == "xiaohongshu_check_login_session"), None)
            if login_status_tool:
                result = await login_status_tool.ainvoke({})
                print(f"   结果: {result}")
            else:
                print("   ⚠️  未找到登录状态检查工具")
        except Exception as e:
            print(f"   ❌ 调用失败: {e}")
        print()
            
    except ImportError as e:
        print("❌ 未安装 langchain-mcp-adapters")
        print("请运行: pip install langchain-mcp-adapters")
        print(f"错误详情: {e}")
        return
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("\n💡 请确保:")
        print("   1. MCP 服务已启动（运行: python -m xiaohongshu_mcp_python.main）")
        print("   2. 服务运行在 http://127.0.0.1:8002")
        print("   3. 如果服务运行在其他地址，请修改 URL")
        return
    
    print("✅ 连接示例完成")
    print()


async def example_connect_with_stdio():
    """
    示例：使用 stdio 传输方式连接（本地进程）
    
    注意：这种方式会启动一个新的 Python 进程来运行 MCP 服务
    """
    print("=" * 80)
    print("示例：使用 stdio 传输连接（本地进程）")
    print("=" * 80)
    print()
    
    try:
        import sys
        from pathlib import Path
        
        # 获取项目根目录
        project_root = Path(__file__).parent.parent
        main_module = project_root / "src" / "xiaohongshu_mcp_python" / "main.py"
        
        if not main_module.exists():
            print(f"❌ 找不到主模块: {main_module}")
            return
        
        print("💡 方式 2: stdio 传输（本地进程）")
        print(f"   模块路径: {main_module}")
        print()
        
        client = MultiServerMCPClient({
            "xiaohongshu": {
                "command": sys.executable,  # 使用当前 Python 解释器
                "args": ["-m", "xiaohongshu_mcp_python.main"],
                "transport": "stdio",
            }
        })
        
        # 连接并获取工具（新 API：直接调用 get_tools()）
        print("📡 正在连接小红书 MCP 服务（stdio）...")
        tools = await client.get_tools()
        print("✅ 成功连接到小红书 MCP 服务（stdio）")
        print()
        
        # 列出所有可用工具
        print(f"📋 可用工具数量: {len(tools)}")
        print()
            
    except ImportError as e:
        print("❌ 未安装 langchain-mcp-adapters")
        print("请运行: pip install langchain-mcp-adapters")
        return
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("✅ stdio 连接示例完成")
    print()




async def example_custom_server_url():
    """
    示例：连接到自定义地址的 MCP 服务
    """
    print("=" * 80)
    print("示例：连接到自定义地址的 MCP 服务")
    print("=" * 80)
    print()
    
    # 自定义服务器地址
    # 如果服务运行在其他地址或端口，修改这里
    custom_host = "127.0.0.1"  # 或 "0.0.0.0" 或其他 IP
    custom_port = 8000         # 或你配置的其他端口
    
    server_url = f"http://{custom_host}:{custom_port}/mcp"
    
    print(f"💡 连接到自定义地址: {server_url}")
    print()
    
    try:
        client = MultiServerMCPClient({
            "xiaohongshu": {
                "url": server_url,
                "transport": "streamable_http",
            }
        })
        
        # 连接并获取工具（新 API：直接调用 get_tools()）
        print(f"📡 正在连接到: {server_url}")
        tools = await client.get_tools()
        print(f"✅ 成功连接到: {server_url}")
        print(f"📋 可用工具: {[tool.name for tool in tools]}")
            
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print(f"\n💡 请确保服务运行在: {server_url}")
        return
    
    print("\n✅ 自定义地址连接示例完成")
    print()


async def main():
    """主函数：运行所有示例"""
    print("\n" + "=" * 80)
    print("小红书 MCP 服务 - langchain_mcp_adapters 连接示例")
    print("=" * 80)
    print()
    
    # 示例 1: HTTP 传输（推荐）
    await example_connect_xiaohongshu_mcp()
    
    # 示例 2: stdio 传输
    # await example_connect_with_stdio()
    
    # 示例 3: 自定义地址
    # await example_custom_server_url()
    
    # 示例 4: 与 LangChain Agent 结合（需要配置 LLM）
    # await example_use_with_langchain_agent()
    
    print("\n" + "=" * 80)
    print("所有示例完成")
    print("=" * 80)


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())

