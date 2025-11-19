"""
StateSnapshot 状态快照演示

演示 graph.get_state() 的作用和 StateSnapshot 对象的属性

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.02-人工介入.state_snapshot_demo
"""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict
from typing import Annotated


# 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]
    counter: int


def create_simple_graph():
    """创建一个简单的图，用于演示状态快照"""
    graph_builder = StateGraph(State)
    
    def node_a(state: State):
        """节点 A：增加计数器"""
        print("[节点 A 执行]")
        return {"counter": state.get("counter", 0) + 1}
    
    def node_b(state: State):
        """节点 B：触发中断"""
        print("[节点 B 执行]")
        print("[节点 B 调用 interrupt，执行暂停]")
        # interrupt() 需要一个 value 参数，这个值会在恢复时通过 Command 返回
        interrupt({"message": "节点 B 请求中断"})  # 中断执行
        return {"counter": state.get("counter", 0) + 1}
    
    def node_c(state: State):
        """节点 C：正常完成"""
        print("[节点 C 执行]")
        return {"counter": state.get("counter", 0) + 1}
    
    # 添加节点
    graph_builder.add_node("node_a", node_a)
    graph_builder.add_node("node_b", node_b)
    graph_builder.add_node("node_c", node_c)
    
    # 添加边
    graph_builder.add_edge(START, "node_a")
    graph_builder.add_edge("node_a", "node_b")
    graph_builder.add_edge("node_b", "node_c")
    graph_builder.add_edge("node_c", END)
    
    # 编译图（启用检查点）
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph


def demonstrate_state_snapshot():
    """演示状态快照的使用"""
    print("=" * 60)
    print("StateSnapshot 状态快照演示")
    print("=" * 60)
    
    graph = create_simple_graph()
    config = {"configurable": {"thread_id": "demo-session"}}
    
    # 初始状态
    initial_state = {
        "messages": [HumanMessage(content="开始执行")],
        "counter": 0
    }
    
    print("\n1. 开始执行图（会被中断）")
    print("-" * 60)
    
    # 执行图（会被中断）
    events = graph.stream(initial_state, config, stream_mode="values")
    for event in events:
        print(f"   收到事件，counter = {event.get('counter', 0)}")
    
    print("\n2. 检查图的状态")
    print("-" * 60)
    
    # 获取状态快照
    snapshot = graph.get_state(config)
    
    print(f"\n状态快照信息:")
    print(f"  - snapshot.values: {snapshot.values}")
    print(f"  - snapshot.next: {snapshot.next}")
    print(f"  - snapshot.tasks: {snapshot.tasks}")
    print(f"  - snapshot.interrupts: {snapshot.interrupts}")
    print(f"  - snapshot.config: {snapshot.config}")
    print(f"  - snapshot.created_at: {snapshot.created_at}")
    
    print("\n3. 分析状态快照")
    print("-" * 60)
    
    # 检查是否被中断
    if snapshot.next:
        print(f"  ⚠️  图被中断了！")
        print(f"  - 下一个要执行的节点: {snapshot.next}")
        print(f"  - 待执行的任务数量: {len(snapshot.tasks)}")
        print(f"  - 中断信息数量: {len(snapshot.interrupts)}")
        
        if snapshot.values:
            print(f"  - 当前 counter 值: {snapshot.values.get('counter', 'N/A')}")
            print(f"  - 当前 messages 数量: {len(snapshot.values.get('messages', []))}")
    else:
        print("  ✅ 图正常完成，没有待执行的节点")
    
    print("\n4. 恢复执行")
    print("-" * 60)
    
    if snapshot.next:
        print("  继续执行图...")
        # 恢复执行：需要使用 Command(resume={...}) 来恢复中断
        # resume 的值会传递给 interrupt() 函数作为返回值
        resume_command = Command(resume={"message": "恢复执行"})
        events = graph.stream(resume_command, config, stream_mode="values")
        for event in events:
            print(f"   收到事件，counter = {event.get('counter', 0)}")
        
        # 再次检查状态
        snapshot_after = graph.get_state(config)
        print(f"\n  执行后的状态:")
        print(f"  - snapshot.next: {snapshot_after.next}")
        if not snapshot_after.next:
            print("  ✅ 图已正常完成")


def demonstrate_state_comparison():
    """演示不同情况下的状态快照"""
    print("\n" + "=" * 60)
    print("不同情况下的状态快照对比")
    print("=" * 60)
    
    graph = create_simple_graph()
    
    # 情况 1：正常完成的图
    print("\n情况 1: 正常完成的图（没有中断）")
    print("-" * 60)
    config1 = {"configurable": {"thread_id": "normal-completion"}}
    
    # 创建一个不会中断的简单图
    def simple_node(state: State):
        return {"counter": 1}
    
    simple_graph_builder = StateGraph(State)
    simple_graph_builder.add_node("simple", simple_node)
    simple_graph_builder.add_edge(START, "simple")
    simple_graph_builder.add_edge("simple", END)
    simple_graph = simple_graph_builder.compile(checkpointer=MemorySaver())
    
    simple_graph.stream(
        {"messages": [], "counter": 0},
        config1,
        stream_mode="values"
    )
    
    snapshot1 = simple_graph.get_state(config1)
    print(f"  snapshot.next: {snapshot1.next}")
    print(f"  结果: {'✅ 正常完成' if not snapshot1.next else '❌ 未完成'}")
    
    # 情况 2：被中断的图
    print("\n情况 2: 被中断的图")
    print("-" * 60)
    config2 = {"configurable": {"thread_id": "interrupted"}}
    
    graph.stream(
        {"messages": [], "counter": 0},
        config2,
        stream_mode="values"
    )
    
    snapshot2 = graph.get_state(config2)
    print(f"  snapshot.next: {snapshot2.next}")
    print(f"  snapshot.tasks: {len(snapshot2.tasks)} 个任务")
    print(f"  snapshot.interrupts: {len(snapshot2.interrupts)} 个中断")
    print(f"  结果: {'⚠️  被中断' if snapshot2.next else '✅ 正常完成'}")


def main():
    """主函数"""
    demonstrate_state_snapshot()
    demonstrate_state_comparison()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
graph.get_state(config) 的作用：

1. 获取图的最新状态快照
   - 从 checkpointer 中读取保存的状态
   - 返回 StateSnapshot 对象

2. StateSnapshot 包含的关键信息：
   - values: 当前状态的所有值（如 messages, counter 等）
   - next: 下一个要执行的节点列表
     * 如果为空 ()，说明图已正常完成
     * 如果有值，说明图被中断或还有待执行的节点
   - tasks: 待执行的任务列表
   - interrupts: 中断信息列表
   - config: 使用的配置
   - metadata: 检查点元数据
   - created_at: 状态创建时间

3. 使用场景：
   - 检查图是否被中断
   - 查看当前状态的值
   - 了解待执行的节点
   - 判断是否需要恢复执行

4. 判断图是否被中断：
   if snapshot.next:
       # 图被中断，需要恢复执行
   else:
       # 图正常完成
    """)


if __name__ == "__main__":
    main()

