"""
个性化助手（综合示例）

本示例展示如何结合配置、状态和自定义提示创建个性化助手。

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.h-上下文.personalized_assistant_example

"""

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from ai_social_scheduler.client import QwenClient


def personalized_prompt(
    state: AgentState,
    config: RunnableConfig,
) -> list[AnyMessage]:
    """个性化提示（根据用户角色和偏好）"""
    user_name = config.get("configurable", {}).get("user_name", "用户")
    user_role = config.get("configurable", {}).get("user_role", "普通用户")
    
    if user_role == "管理员":
        system_msg = f"你是 {user_name} 的专属AI助手。你拥有管理员权限，可以提供更高级的功能。"
    else:
        system_msg = f"你是 {user_name} 的专属AI助手。请友好地帮助用户解决问题。"
    
    return [{"role": "system", "content": system_msg}] + state["messages"]


@tool
def admin_tool(config: RunnableConfig) -> str:
    """管理员工具（仅管理员可用）"""
    user_role = config.get("configurable", {}).get("user_role", "普通用户")
    if user_role == "管理员":
        return "管理员功能：可以访问系统管理功能"
    else:
        return "权限不足：此功能仅管理员可用"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    weather_data = {
        "北京": "晴朗，温度 25°C",
        "上海": "多云，温度 22°C",
        "深圳": "晴朗，温度 28°C",
    }
    return f"{city} 的天气：{weather_data.get(city, '晴朗，温度 25°C')}"


def main():
    """示例：个性化助手（综合示例）"""
    print("=" * 80)
    print("示例：个性化助手（综合示例）")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        qwen_client.client,
        tools=[admin_tool, get_weather],
        prompt=personalized_prompt,
    )
    
    # 测试普通用户
    print("💡 测试普通用户...")
    response1 = agent.invoke(
        {"messages": [HumanMessage(content="使用管理员工具")]},
        config={"configurable": {"user_name": "赵六", "user_role": "普通用户"}}
    )
    
    if "messages" in response1 and response1["messages"]:
        last_message = response1["messages"][-1]
        print(f"用户角色: 普通用户")
        print(f"回答: {last_message.content}")
    
    print()
    
    # 测试管理员
    print("💡 测试管理员...")
    response2 = agent.invoke(
        {"messages": [HumanMessage(content="使用管理员工具")]},
        config={"configurable": {"user_name": "管理员", "user_role": "管理员"}}
    )
    
    if "messages" in response2 and response2["messages"]:
        last_message = response2["messages"][-1]
        print(f"用户角色: 管理员")
        print(f"回答: {last_message.content}")
    
    print()
    print("✅ 个性化助手示例完成")
    print()


if __name__ == "__main__":
    main()

