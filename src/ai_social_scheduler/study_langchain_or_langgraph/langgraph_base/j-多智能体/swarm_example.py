"""
群组多智能体系统示例

本示例展示如何使用 langgraph-swarm 创建群组多智能体系统：
1. 创建多个专业智能体（航班预订助手、酒店预订助手）
2. 为每个智能体创建移交工具，使其能够将控制权移交给其他智能体
3. 智能体根据其专业性动态地相互移交控制权

群组架构特点：
- 智能体根据其专业性动态地相互移交控制权
- 系统会记住哪个智能体上次处于活动状态
- 在后续交互中，对话会与该智能体恢复
- 没有中央主管，智能体之间直接通信

================================================================================
Swarm（群组）架构适合什么场景？
================================================================================

✅ 适合的场景：

1. **专业领域协作场景**
   - 每个智能体都是某个领域的专家
   - 任务可能涉及多个领域，需要智能体之间协作
   - 例如：旅行规划（航班、酒店、租车）、医疗咨询（内科、外科、药房）

2. **去中心化决策场景**
   - 不需要中央协调者，智能体可以自主决定何时移交
   - 智能体之间是平等的，没有层级关系
   - 例如：客服系统（技术支持、销售、售后）

3. **上下文连续性场景**
   - 需要记住上次与哪个智能体交互
   - 用户可能多次返回，希望继续之前的对话
   - 例如：多轮对话系统、长期客户服务

4. **灵活的任务流转场景**
   - 任务边界不清晰，可能需要在多个智能体间流转
   - 智能体可以根据当前情况决定下一步
   - 例如：复杂问题解决、多步骤工作流

5. **专业工具集场景**
   - 每个智能体有自己独特的工具集
   - 工具使用需要专业知识判断
   - 例如：数据分析（不同分析工具）、内容创作（不同创作工具）

❌ 不适合的场景：

1. **需要严格任务分配的场景**
   - 任务必须按照固定流程执行
   - 需要中央协调者统一调度
   - → 更适合使用 Supervisor（主管）架构

2. **简单单领域任务**
   - 任务只涉及单一领域
   - 不需要多个智能体协作
   - → 使用单个智能体即可

3. **需要全局状态管理的场景**
   - 需要中央状态管理
   - 需要全局决策和协调
   - → Supervisor 架构更适合

================================================================================
Swarm vs Supervisor 对比
================================================================================

Swarm（群组）：
- 架构：去中心化，智能体之间直接通信
- 决策：每个智能体自主决定何时移交
- 适用：专业协作、灵活流转、上下文连续性
- 优势：灵活、自主、适合复杂协作
- 劣势：可能产生循环移交、缺乏全局视角

Supervisor（主管）：
- 架构：中心化，主管统一协调
- 决策：主管根据任务分配给智能体
- 适用：严格流程、全局协调、任务分配
- 优势：统一管理、避免循环、全局视角
- 劣势：主管成为瓶颈、不够灵活

================================================================================

安装依赖：
pip install langgraph-swarm

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.j-多智能体.swarm_example

"""

import json

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from ai_social_scheduler.client import QwenClient


# ==================== 工具定义 ====================

@tool
def book_hotel(hotel_name: str) -> str:
    """预订酒店"""
    return f"成功预订酒店: {hotel_name}"


@tool
def book_flight(from_airport: str, to_airport: str) -> str:
    """预订航班"""
    return f"成功预订航班: 从 {from_airport} 到 {to_airport}"


# ==================== 创建智能体 ====================

def flight_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    """
    航班预订助手的提示
    
    注意：长时间运行时，如果消息太多，可以考虑实现上下文管理：
    - 使用滑动窗口：只保留最近 N 条消息
    - 使用消息摘要：将旧消息压缩为摘要
    - 使用长期记忆：将重要信息存储到外部，只检索相关内容
    """
    system_msg = (
        "你是一个航班预订助手，专门帮助用户预订航班。"
        "重要：当你需要处理航班预订时，先完成航班预订任务，然后再考虑是否需要移交。"
        "如果需要预订酒店，在完成航班预订后，使用 transfer_to_hotel_assistant 工具将用户移交给酒店预订助手。"
        "不要同时调用多个工具，先完成当前任务再移交。"
    )
    
    messages = state["messages"]
    
    # 可选：如果消息太多，可以截取最近的消息（防止上下文爆炸）
    # 取消下面的注释来启用滑动窗口
    # max_context_messages = 30  # 保留最近 30 条消息
    # if len(messages) > max_context_messages:
    #     # 保留系统消息（如果有）和最近的消息
    #     system_msgs = [msg for msg in messages if isinstance(msg, dict) and msg.get("type") == "system"]
    #     other_msgs = [msg for msg in messages if not (isinstance(msg, dict) and msg.get("type") == "system")]
    #     recent_msgs = other_msgs[-max_context_messages:]
    #     messages = system_msgs + recent_msgs
    
    return [{"role": "system", "content": system_msg}] + messages


def hotel_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    """
    酒店预订助手的提示
    
    注意：长时间运行时，如果消息太多，可以考虑实现上下文管理
    """
    system_msg = (
        "你是一个酒店预订助手，专门帮助用户预订酒店。"
        "重要：当你需要处理酒店预订时，先完成酒店预订任务，然后再考虑是否需要移交。"
        "如果需要预订航班，在完成酒店预订后，使用 transfer_to_flight_assistant 工具将用户移交给航班预订助手。"
        "不要同时调用多个工具，先完成当前任务再移交。"
    )
    
    messages = state["messages"]
    
    # 可选：如果消息太多，可以截取最近的消息（防止上下文爆炸）
    # 取消下面的注释来启用滑动窗口
    # max_context_messages = 30  # 保留最近 30 条消息
    # if len(messages) > max_context_messages:
    #     system_msgs = [msg for msg in messages if isinstance(msg, dict) and msg.get("type") == "system"]
    #     other_msgs = [msg for msg in messages if not (isinstance(msg, dict) and msg.get("type") == "system")]
    #     recent_msgs = other_msgs[-max_context_messages:]
    #     messages = system_msgs + recent_msgs
    
    return [{"role": "system", "content": system_msg}] + messages


def create_agents_with_handoff():
    """创建带有移交工具的智能体"""
    try:
        from langgraph_swarm import create_handoff_tool
    except ImportError:
        print("❌ 未安装 langgraph-swarm")
        print("请运行: pip install langgraph-swarm")
        return None, None
    
    qwen_client = QwenClient(model="qwen-plus", temperature=0.7)
    
    # 创建移交工具
    # 这些工具允许智能体将控制权移交给其他智能体
    transfer_to_hotel_assistant = create_handoff_tool(
        agent_name="hotel_assistant",
        description="将用户移交给酒店预订助手。当用户需要预订酒店时使用此工具。",
    )
    
    transfer_to_flight_assistant = create_handoff_tool(
        agent_name="flight_assistant",
        description="将用户移交给航班预订助手。当用户需要预订航班时使用此工具。",
    )
    
    # 航班预订助手
    # 包含航班预订工具和移交到酒店助手的工具
    # 注意: create_react_agent 虽然已弃用，但新 API (create_agent) 的参数不同，暂时继续使用
    flight_assistant = create_react_agent(
        qwen_client.client,
        tools=[book_flight, transfer_to_hotel_assistant],
        prompt=flight_prompt,
        name="flight_assistant",
    )
    
    # 酒店预订助手
    # 包含酒店预订工具和移交到航班助手的工具
    hotel_assistant = create_react_agent(
        qwen_client.client,
        tools=[book_hotel, transfer_to_flight_assistant],
        prompt=hotel_prompt,
        name="hotel_assistant",
    )
    
    return flight_assistant, hotel_assistant


# ==================== 创建群组 ====================

def create_swarm_system():
    """
    创建群组多智能体系统
    
    Swarm 记忆机制说明：
    ========================================================================
    
    1. **活动智能体记忆（Active Agent Memory）**
       - Swarm 会在状态中保存 `active_agent` 字段
       - 记录当前哪个智能体处于活动状态
       - 当发生移交时，`active_agent` 会更新为目标智能体
       - 下次对话时，系统会从上次活动的智能体开始
    
    2. **消息历史记忆（Message History）**
       - 使用 LangGraph 的 checkpoint 机制保存完整的消息历史
       - 所有智能体的对话历史都会被保存
       - 通过 `thread_id` 来区分不同的会话
    
    3. **状态持久化**
       - 默认使用内存保存（InMemorySaver）
       - 可以配置使用其他 checkpointer（如数据库）
       - 状态包括：
         * 当前活动智能体（active_agent）
         * 完整的消息历史（messages）
         * 智能体之间的移交历史
    
    4. **记忆恢复机制**
       - 使用相同的 `thread_id` 可以恢复之前的对话
       - 系统会从上次活动的智能体继续对话
       - 所有历史消息都会被传递给当前活动的智能体
    
    5. **配置 checkpointer（可选）**
       - 默认情况下，swarm 使用内存保存
       - 如果需要持久化，可以传入 checkpointer：
         
         from langgraph.checkpoint.memory import MemorySaver
         checkpointer = MemorySaver()
         
         swarm = create_swarm(
             agents=[...],
             default_active_agent="...",
         ).compile(checkpointer=checkpointer)
    
    6. **使用 thread_id 区分会话**
       - 每次调用时，可以通过 config 传入 thread_id：
         
         swarm.invoke(
             {"messages": [...]},
             config={"configurable": {"thread_id": "user_123"}}
         )
       - 不同的 thread_id 会有独立的记忆空间
    
    7. **⚠️ 长时间运行的上下文处理（重要！）**
       
       LangGraph 的默认行为：
       - ❌ 默认情况下，LangGraph 会将**所有历史消息**都传递给 LLM
       - ❌ 不会自动截断或压缩上下文
       - ❌ 不会自动管理 token 数量
       
       可能遇到的问题：
       - 🔴 Token 数量不断增长，可能超过模型的上下文窗口限制（如 32K、128K）
       - 🔴 API 调用成本增加（按 token 计费，输入 token 越多越贵）
       - 🔴 响应速度变慢（处理更多 token 需要更长时间）
       - 🔴 内存占用增加（保存所有历史消息）
       - 🔴 可能导致 API 调用失败（超过上下文窗口）
       
       解决方案：
       
       a) **在 prompt 函数中实现滑动窗口（推荐）**
          - 在调用 LLM 前，截取最近 N 条消息
          - 丢弃旧消息，只保留最新的上下文
          - 示例代码：
            
            def flight_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
                system_msg = "..."
                messages = state["messages"]
                
                # 只保留最近 30 条消息（防止上下文爆炸）
                max_messages = 30
                if len(messages) > max_messages:
                    messages = messages[-max_messages:]
                
                return [{"role": "system", "content": system_msg}] + messages
       
       b) **消息摘要（Message Summarization）**
          - 定期将旧消息压缩为摘要
          - 保留摘要 + 最近的消息
          - 使用 LLM 生成摘要，保留关键信息
          - 适合需要保留长期上下文的场景
       
       c) **长期记忆存储（Long-term Memory）**
          - 将重要信息存储到外部存储（数据库、向量数据库）
          - 使用 RAG 检索相关历史
          - 只将检索到的相关信息放入上下文
          - 适合需要跨会话记忆的场景
       
       d) **智能消息过滤**
          - 根据相关性过滤消息
          - 只保留与当前任务相关的历史
          - 使用语义搜索找到相关消息
       
       e) **分段对话**
          - 定期创建新的 thread_id
          - 将重要信息提取到长期记忆
          - 新对话从长期记忆恢复上下文
       
       推荐方案组合：
       1. 短期（< 50 条消息）：不做处理，保留全部
       2. 中期（50-200 条消息）：滑动窗口（保留最近 30-50 条）
       3. 长期（> 200 条消息）：消息摘要 + 长期记忆存储
       
       注意：
       - Swarm 模式下，每个智能体的 prompt 函数都会收到完整的消息历史
       - 需要在每个智能体的 prompt 函数中都实现上下文管理
       - 或者创建一个统一的上下文管理函数，在所有 prompt 函数中调用
    
    ========================================================================
    """
    try:
        from langgraph_swarm import create_swarm
    except ImportError:
        print("❌ 未安装 langgraph-swarm")
        print("请运行: pip install langgraph-swarm")
        return None
    
    # 创建带有移交工具的智能体
    flight_assistant, hotel_assistant = create_agents_with_handoff()
    if flight_assistant is None or hotel_assistant is None:
        return None
    
    # 创建群组系统
    # default_active_agent 指定默认活动智能体（首次对话时启动的智能体）
    # 注意：如果使用相同的 thread_id，系统会记住上次活动的智能体
    #       并从那开始，而不是从 default_active_agent 开始
    swarm = create_swarm(
        agents=[flight_assistant, hotel_assistant],
        default_active_agent="flight_assistant",
    ).compile()
    
    # 可选：如果需要持久化记忆，可以传入 checkpointer
    # from langgraph.checkpoint.memory import MemorySaver
    # checkpointer = MemorySaver()
    # swarm = create_swarm(
    #     agents=[flight_assistant, hotel_assistant],
    #     default_active_agent="flight_assistant",
    # ).compile(checkpointer=checkpointer)
    
    return swarm


# ==================== 格式化输出 ====================

def detect_handoff(chunk: dict) -> tuple[bool, str | None, str | None]:
    """
    检测 chunk 中是否有移交事件
    
    Args:
        chunk: 群组系统返回的块数据
    
    Returns:
        (是否有移交, 源智能体, 目标智能体)
    """
    # 先将 chunk 转换为字典格式以便检查
    def to_dict_safe(obj):
        """安全地将对象转换为字典"""
        if isinstance(obj, dict):
            return obj
        elif hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return obj
    
    chunk_dict = to_dict_safe(chunk)
    
    for agent_name, agent_data in chunk_dict.items():
        agent_data = to_dict_safe(agent_data)
        
        # 检查是否有 active_agent 字段（表示移交）
        if isinstance(agent_data, dict) and "active_agent" in agent_data:
            target_agent = agent_data["active_agent"]
            source_agent = agent_name
            return True, source_agent, target_agent
        
        # 检查消息中是否有移交工具调用
        if isinstance(agent_data, dict) and "messages" in agent_data:
            messages = agent_data["messages"]
            for msg in messages:
                msg_dict = to_dict_safe(msg)
                
                # 检查是否是工具消息，且是移交工具
                if isinstance(msg_dict, dict):
                    msg_type = msg_dict.get("type") or (hasattr(msg, 'type') and getattr(msg, 'type', None))
                    tool_name = msg_dict.get("name") or (hasattr(msg, 'name') and getattr(msg, 'name', ''))
                    
                    if msg_type == "tool" and tool_name and "transfer_to" in str(tool_name):
                        source_agent = agent_name
                        # 从工具名提取目标智能体名
                        # transfer_to_hotel_assistant -> hotel_assistant
                        if "transfer_to_" in str(tool_name):
                            target_agent = str(tool_name).replace("transfer_to_", "")
                            return True, source_agent, target_agent
                    
                    # 检查 AI 消息中的工具调用
                    elif msg_type == "ai":
                        tool_calls = msg_dict.get("tool_calls", [])
                        if hasattr(msg, 'tool_calls'):
                            tool_calls = [to_dict_safe(tc) for tc in msg.tool_calls] if msg.tool_calls else []
                        
                        for tool_call in tool_calls:
                            tool_call_dict = to_dict_safe(tool_call)
                            tool_name = tool_call_dict.get("name", "")
                            if "transfer_to" in str(tool_name):
                                source_agent = agent_name
                                if "transfer_to_" in str(tool_name):
                                    target_agent = str(tool_name).replace("transfer_to_", "")
                                    return True, source_agent, target_agent
    
    return False, None, None


def format_chunk(chunk: dict) -> str:
    """
    格式化 chunk 输出为易读的 JSON 格式
    
    Args:
        chunk: 群组系统返回的块数据
    
    Returns:
        格式化后的 JSON 字符串
    """
    # 将消息对象转换为字典格式以便序列化
    def convert_to_dict(obj):
        # 优先使用 model_dump() (Pydantic v2)
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        # 其次使用 dict() (Pydantic v1)
        elif hasattr(obj, 'dict'):
            return obj.dict()
        # 对于普通对象，使用 __dict__
        elif hasattr(obj, '__dict__'):
            result = {}
            for k, v in obj.__dict__.items():
                # 跳过私有属性和特殊属性
                if not k.startswith('_'):
                    result[k] = convert_to_dict(v)
            return result
        # 处理字典
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        # 处理列表和元组
        elif isinstance(obj, (list, tuple)):
            return [convert_to_dict(item) for item in obj]
        # 其他基本类型直接返回
        else:
            return obj
    
    # 转换整个 chunk
    serializable_chunk = convert_to_dict(chunk)
    
    # 使用 json.dumps 格式化输出
    return json.dumps(serializable_chunk, indent=2, ensure_ascii=False)


# ==================== 示例：使用群组系统 ====================

def example_swarm_system():
    """示例：使用群组多智能体系统"""
    print("=" * 80)
    print("示例：群组多智能体系统")
    print("=" * 80)
    print()
    
    swarm = create_swarm_system()
    if swarm is None:
        return
    
    print("💡 群组系统已创建")
    print("   智能体列表:")
    print("   - flight_assistant: 航班预订助手（默认活动）")
    print("   - hotel_assistant: 酒店预订助手")
    print()
    print("   特点:")
    print("   - 智能体可以动态地将控制权移交给其他智能体")
    print("   - 系统会记住上次活动的智能体")
    print("   - 没有中央主管，智能体之间直接通信")
    print()
    
    print("📝 运行群组系统...")
    print("   任务 1: 预订从北京到上海的航班")
    print("-" * 80)
    
    # 使用群组系统处理第一个任务
    for chunk in swarm.stream(
        {
            "messages": [
                HumanMessage(content="预订从北京到上海的航班")
            ]
        }
    ):
        # 检测移交事件
        has_handoff, source, target = detect_handoff(chunk)
        if has_handoff:
            source_display = {
                "flight_assistant": "✈️ 航班助手",
                "hotel_assistant": "🏨 酒店助手"
            }.get(source, source)
            target_display = {
                "flight_assistant": "✈️ 航班助手",
                "hotel_assistant": "🏨 酒店助手"
            }.get(target, target)
            print(f"🔄 移交事件: {source_display} → {target_display}")
            print()
        
        print(format_chunk(chunk))
        print()
    
    print()
    print("   任务 2: 预订一家名为 McKittrick 的酒店")
    print("-" * 80)
    
    # 使用群组系统处理第二个任务（演示系统会记住上次活动的智能体）
    for chunk in swarm.stream(
        {
            "messages": [
                HumanMessage(content="预订一家名为 McKittrick 的酒店，和北京到上海的航班")
            ]
        }
    ):
        # 检测移交事件
        has_handoff, source, target = detect_handoff(chunk)
        if has_handoff:
            source_display = {
                "flight_assistant": "✈️ 航班助手",
                "hotel_assistant": "🏨 酒店助手"
            }.get(source, source)
            target_display = {
                "flight_assistant": "✈️ 航班助手",
                "hotel_assistant": "🏨 酒店助手"
            }.get(target, target)
            print(f"🔄 移交事件: {source_display} → {target_display}")
            print()
        
        print(format_chunk(chunk))
        print()
    
    print()
    print("✅ 群组系统示例完成")
    print()


# ==================== 示例：演示记忆机制 ====================

def example_swarm_memory():
    """示例：演示 Swarm 的记忆机制"""
    print("=" * 80)
    print("示例：Swarm 记忆机制演示")
    print("=" * 80)
    print()
    
    swarm = create_swarm_system()
    if swarm is None:
        return
    
    # 使用相同的 thread_id 来演示记忆功能
    thread_id = "memory_demo_session"
    
    print("💡 记忆机制演示")
    print("   使用相同的 thread_id 来保持对话记忆")
    print(f"   thread_id: {thread_id}")
    print()
    
    # 第一次对话：从默认智能体开始
    print("📝 第一次对话：预订航班")
    print("-" * 80)
    response1 = swarm.invoke(
        {"messages": [HumanMessage(content="预订从北京到上海的航班")]},
        config={"configurable": {"thread_id": thread_id}}
    )
    print("✅ 第一次对话完成")
    print()
    
    # 第二次对话：系统会记住上次活动的智能体
    print("📝 第二次对话：继续预订（系统会记住上次的智能体）")
    print("-" * 80)
    response2 = swarm.invoke(
        {"messages": [HumanMessage(content="再帮我预订一家酒店")]},
        config={"configurable": {"thread_id": thread_id}}
    )
    print("✅ 第二次对话完成")
    print()
    
    # 说明记忆机制
    print("💡 记忆机制说明：")
    print("   1. 第一次对话后，系统记住了当前活动的智能体")
    print("   2. 第二次对话时，系统从上次活动的智能体继续")
    print("   3. 所有历史消息都被保留，智能体可以看到完整的对话历史")
    print("   4. 如果发生移交，系统会更新 active_agent 字段")
    print()


# ==================== 主函数 ====================

def main():
    """运行群组多智能体系统示例"""
    try:
        example_swarm_system()
    except ValueError as e:
        print(f"\n❌ 配置错误: {e}")
        print("请确保在 .env 文件中配置了 ALIBABA_BAILIAN_API_KEY")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
 