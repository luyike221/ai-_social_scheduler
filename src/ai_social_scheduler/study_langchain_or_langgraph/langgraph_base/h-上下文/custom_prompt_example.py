"""
使用上下文自定义提示示例

本示例展示如何根据配置或状态动态生成提示。

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.h-上下文.custom_prompt_example

"""

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from ai_social_scheduler.client import QwenClient


def custom_prompt(
    state: AgentState,
    config: RunnableConfig,
) -> list[AnyMessage]:
    """
    自定义提示函数（根据上下文动态生成）
    
    原理和机制：
    1. LangGraph 在每次调用 LLM 之前会调用这个函数
    2. 函数接收两个参数：
       - state: 当前状态（包含 messages 等）
       - config: 运行时配置（包含 configurable 中的静态数据）
    3. 函数返回消息列表，LangGraph 会将这个列表传递给 LLM
    4. 通过动态生成 system message，可以实现：
       - 个性化提示（根据用户信息）
       - 条件行为（根据用户角色）
       - 上下文感知（根据状态信息）
    
    执行流程：
    invoke() -> LangGraph -> custom_prompt(state, config) -> 生成消息列表 -> 传递给 LLM
    """
    user_name = config.get("configurable", {}).get("user_name", "用户")
    system_msg = f"你是一个有用的AI助手。当前用户的名字是 {user_name}。请用友好的方式与用户交流。"
    
    # 返回的消息列表会传递给 LLM
    # system message 在最前面，然后是用户消息
    return [{"role": "system", "content": system_msg}] + state["messages"]


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
    """示例：使用上下文自定义提示"""
    print("=" * 80)
    print("示例：使用上下文自定义提示")
    print("=" * 80)
    print()
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    agent = create_react_agent(
        qwen_client.client,
        tools=[get_weather],
        prompt=custom_prompt,  # 传入自定义提示函数
    )
    
    print("💡 使用自定义提示（根据用户名称）...")
    # 机制说明：
    # 1. invoke 时传入 config，包含 user_name
    # 2. LangGraph 每次调用 LLM 前会调用 custom_prompt(state, config)
    # 3. custom_prompt 从 config 中读取 user_name，动态生成 system message
    # 4. 生成的提示包含个性化信息，影响 LLM 的行为
    response = agent.invoke(
        {"messages": [HumanMessage(content="查询北京的天气")]},
        config={"configurable": {"user_name": "王五"}}  # 配置数据会传递给 custom_prompt
    )
    
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        print(f"问题: 查询北京的天气")
        print(f"回答: {last_message.content}")
    
    print()
    print("✅ 自定义提示示例完成")
    print()


if __name__ == "__main__":
    main()

