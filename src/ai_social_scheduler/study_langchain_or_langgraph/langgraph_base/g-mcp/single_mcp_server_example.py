"""
单个 MCP 服务器示例

本示例展示如何使用单个 MCP 服务器：
1. 使用 stdio 传输（本地 Python 脚本）
2. 使用 HTTP 传输（远程服务器）

安装依赖：
pip install langchain-mcp-adapters

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.g-mcp.single_mcp_server_example

"""

import asyncio

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from ai_social_scheduler.client import QwenClient
from ai_social_scheduler.config.model_config import model_config


async def example_single_mcp_server():
    """示例: 使用单个 MCP 服务器 - 阿里云百炼高德地图服务"""
    print("=" * 80)
    print("示例: 使用单个 MCP 服务器 - 高德地图服务")
    print("=" * 80)
    print()
    
    # 从配置中获取 API Key
    alibaba_config = model_config.get_alibaba_bailian_config()
    api_key = alibaba_config.api_key
    
    # 配置阿里云百炼高德地图 MCP 服务器
    client = MultiServerMCPClient({
        "amap-maps": {
            "url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
            "transport": "streamable_http",
            "headers": {
                "Authorization": f"Bearer {api_key}"
            }
        }
    })
    
    print("📡 连接 MCP 服务器...")
    tools = await client.get_tools()
    print(f"✅ 获取到 {len(tools)} 个工具")
    print()
    
    # 显示可用工具
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    print()
    
    # 创建代理
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    agent = create_react_agent(
        qwen_client.client,
        tools
    )
    
    print("🤖 代理已创建，可以开始使用高德地图功能")
    print("   支持功能：地理编码、逆地理编码、IP定位、天气查询、路径规划等")
    print()
    print("💡 使用示例:")
    print("   response = await agent.ainvoke({")
    print('       "messages": [HumanMessage(content="查询北京的天气")]')
    print("   })")
    print()
    
    # 实际调用示例（可以取消注释来测试）
    # print("🗺️  测试高德地图功能...")
    # response = await agent.ainvoke({
    #     "messages": [HumanMessage(content="查询北京的天气")]
    # })
    # 
    # if "messages" in response and response["messages"]:
    #     last_message = response["messages"][-1]
    #     print(f"回答: {last_message.content}")
    
    print()
    print("✅ 单个 MCP 服务器示例完成")
    print()


async def main():
    """主函数"""
    try:
        await example_single_mcp_server()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

