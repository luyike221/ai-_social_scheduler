"""
生成器（Generator）机制演示

这个文件演示了 Python 生成器的工作原理，帮助理解 graph.stream() 的迭代器机制。

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.generator_mechanism_demo
"""


def simple_generator():
    """
    简单的生成器示例
    
    生成器函数使用 yield 关键字，而不是 return
    每次调用 next() 或迭代时，会执行到下一个 yield 并暂停
    """
    print("生成器开始执行...")
    yield "第一个事件"
    print("生成器继续执行...")
    yield "第二个事件"
    print("生成器继续执行...")
    yield "第三个事件"
    print("生成器执行完成")


def simulate_graph_stream():
    """
    模拟 graph.stream() 的行为
    
    这个函数模拟了 LangGraph 的 stream 方法如何工作：
    1. 返回一个生成器，而不是立即执行
    2. 只有在迭代时才会执行
    3. 每个节点执行后 yield 一个事件
    4. 如果被中断，生成器会停止
    """
    print("\n" + "=" * 60)
    print("模拟 graph.stream() 的行为")
    print("=" * 60)
    
    def graph_stream_simulation():
        """模拟图的执行流程"""
        print("[图开始执行]")
        yield {"node": "chatbot", "status": "执行中"}
        print("[chatbot 节点完成]")
        yield {"node": "tools", "status": "执行中"}
        print("[tools 节点执行，遇到 interrupt，中断...]")
        # 模拟中断：生成器在这里停止
        return  # 生成器结束
    
    # 调用 graph.stream() - 注意：图还没有执行！
    print("\n1. 调用 graph.stream() - 返回生成器对象")
    events = graph_stream_simulation()
    print(f"   events 的类型: {type(events)}")
    print(f"   events 的值: {events}")
    print("   ⚠️  注意：图还没有执行！")
    
    # 开始迭代 - 这时图才开始执行
    print("\n2. 开始 for 循环迭代 - 图开始执行")
    print("-" * 60)
    for event in events:
        print(f"   收到事件: {event}")
        print(f"   处理事件...")
    print("-" * 60)
    print("   for 循环结束 - 生成器已耗尽")
    
    # 尝试再次迭代 - 不会工作！
    print("\n3. 尝试再次迭代同一个生成器")
    print("   尝试迭代...")
    count = 0
    for event in events:
        count += 1
        print(f"   收到事件: {event}")
    print(f"   结果: 没有收到任何事件（生成器已耗尽）")


def demonstrate_interrupt_mechanism():
    """
    演示中断机制
    
    模拟 LangGraph 的中断和恢复机制
    """
    print("\n" + "=" * 60)
    print("演示中断和恢复机制")
    print("=" * 60)
    
    # 第一次执行（被中断）
    def first_stream():
        """第一次执行，会被中断"""
        print("[第一次执行开始]")
        yield {"step": 1, "node": "chatbot", "message": "LLM 响应"}
        yield {"step": 2, "node": "tools", "message": "调用工具"}
        print("[遇到 interrupt，执行中断]")
        # 模拟中断：生成器在这里停止
        return
    
    print("\n第一次执行（会被中断）:")
    print("-" * 60)
    events1 = first_stream()
    for event in events1:
        print(f"  收到事件: {event}")
    print("  第一次执行结束（被中断）")
    print("-" * 60)
    
    # 模拟保存状态
    print("\n[状态已保存到 checkpointer]")
    
    # 恢复执行（新的生成器）
    def resume_stream():
        """恢复执行，从上次中断的地方继续"""
        print("[恢复执行，从上次中断的地方继续]")
        yield {"step": 3, "node": "tools", "message": "工具执行完成"}
        yield {"step": 4, "node": "chatbot", "message": "最终响应"}
        print("[执行完成]")
    
    print("\n恢复执行（新的生成器）:")
    print("-" * 60)
    events2 = resume_stream()  # 注意：这是一个全新的生成器！
    for event in events2:
        print(f"  收到事件: {event}")
    print("  恢复执行完成")
    print("-" * 60)
    
    print("\n关键点:")
    print("  - events1 和 events2 是两个不同的生成器")
    print("  - events1 已经耗尽，不能再次使用")
    print("  - events2 是新的生成器，包含恢复执行后的事件")


def demonstrate_lazy_evaluation():
    """
    演示惰性求值（Lazy Evaluation）
    
    生成器是"惰性"的，只有在需要时才执行
    """
    print("\n" + "=" * 60)
    print("演示惰性求值")
    print("=" * 60)
    
    def lazy_generator():
        """惰性生成器"""
        print("  [生成器函数被调用]")
        print("  [但代码还没有执行]")
        yield "第一个值"
        print("  [现在执行到这里]")
        yield "第二个值"
        print("  [继续执行]")
        yield "第三个值"
        print("  [执行完成]")
    
    print("\n1. 调用生成器函数")
    gen = lazy_generator()
    print("   生成器对象已创建，但代码还没有执行")
    
    print("\n2. 调用 next() 获取第一个值")
    first_value = next(gen)
    print(f"   获取到: {first_value}")
    
    print("\n3. 再次调用 next() 获取第二个值")
    second_value = next(gen)
    print(f"   获取到: {second_value}")
    
    print("\n4. 使用 for 循环获取剩余的值")
    for value in gen:
        print(f"   获取到: {value}")


def main():
    """主函数"""
    print("=" * 60)
    print("生成器（Generator）机制演示")
    print("=" * 60)
    
    # 1. 简单生成器示例
    print("\n" + "=" * 60)
    print("1. 简单生成器示例")
    print("=" * 60)
    gen = simple_generator()
    print("生成器对象已创建")
    print("\n开始迭代:")
    for item in gen:
        print(f"  收到: {item}")
    
    # 2. 模拟 graph.stream()
    simulate_graph_stream()
    
    # 3. 演示中断机制
    demonstrate_interrupt_mechanism()
    
    # 4. 演示惰性求值
    demonstrate_lazy_evaluation()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
生成器的关键特性：

1. 惰性执行（Lazy Evaluation）
   - 只有在迭代时才会执行
   - 调用生成器函数不会立即执行代码

2. 一次性使用（Single Use）
   - 生成器只能迭代一次
   - 迭代完成后，生成器就耗尽了
   - 不能重复使用同一个生成器

3. 实时产生（Real-time Generation）
   - 事件在执行过程中逐步产生
   - 不需要等待所有事件都产生完

4. 中断和恢复
   - 如果执行被中断，生成器会停止
   - 需要创建新的生成器来恢复执行
   - 新生成器会从保存的状态继续

graph.stream() 的工作原理：
  - 返回一个生成器对象
  - 只有在 for 循环迭代时，图才开始执行
  - 每个节点执行后 yield 一个事件
  - 如果被中断，生成器停止，需要重新调用 stream() 创建新生成器
    """)


if __name__ == "__main__":
    main()

