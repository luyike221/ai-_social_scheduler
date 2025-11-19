"""
Command 对象如何恢复执行 - 详细演示

这个文件详细解释 Command 和 interrupt 的配合机制

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.command_resume_demo
"""


def explain_interrupt_command_mechanism():
    """解释 interrupt 和 Command 的配合机制"""
    print("=" * 60)
    print("interrupt() 和 Command 的配合机制")
    print("=" * 60)
    
    print("""
核心机制：

1. interrupt() 的双重行为：
   ┌─────────────────────────────────────────┐
   │ 第一次调用（执行被中断）                │
   └─────────────────────────────────────────┘
   - interrupt(value) 被调用
   - 抛出一个 GraphInterrupt 异常
   - 图的执行被暂停
   - 状态被保存到 checkpointer
   - 传入的 value 被保存，用于向客户端展示
   - 函数不会返回（因为抛出了异常）
   
   ┌─────────────────────────────────────────┐
   │ 恢复执行时（使用 Command）               │
   └─────────────────────────────────────────┘
   - 节点从开始重新执行（re-execute）
   - interrupt() 再次被调用
   - LangGraph 检测到有 Command(resume={...})
   - interrupt() 不再抛出异常
   - 直接返回 Command.resume 的值
   - 节点继续执行，就像从未被中断一样

2. Command 对象的作用：
   Command(resume={...}) 告诉 LangGraph：
   - 有一个中断需要恢复
   - 恢复时，interrupt() 应该返回什么值
   - 从哪个中断恢复（如果有多个中断）

3. 执行流程对比：

   第一次执行（被中断）：
   ┌─────────────────────────────────────┐
   │ tools 节点开始执行                   │
   │   ↓                                  │
   │ human_assistance 工具被调用          │
   │   ↓                                  │
   │ interrupt({"query": "..."}) 被调用   │
   │   ↓                                  │
   │ 抛出 GraphInterrupt 异常             │
   │   ↓                                  │
   │ 执行暂停                             │
   │ 状态保存到 checkpointer              │
   │ 生成器结束                           │
   └─────────────────────────────────────┘

   恢复执行：
   ┌─────────────────────────────────────┐
   │ Command(resume={"data": "..."})     │
   │   ↓                                  │
   │ graph.stream(human_command, config) │
   │   ↓                                  │
   │ 从 checkpointer 读取状态            │
   │   ↓                                  │
   │ tools 节点重新开始执行               │
   │   ↓                                  │
   │ human_assistance 工具再次被调用      │
   │   ↓                                  │
   │ interrupt() 再次被调用               │
   │   ↓                                  │
   │ 检测到有 resume 值                   │
   │   ↓                                  │
   │ 返回 resume 值（不抛异常）           │
   │   ↓                                  │
   │ human_assistance 返回结果             │
   │   ↓                                  │
   │ tools 节点完成                        │
   │   ↓                                  │
   │ 继续执行后续节点                      │
   └─────────────────────────────────────┘
    """)


def explain_resume_parameter():
    """解释 resume 参数的不同格式"""
    print("\n" + "=" * 60)
    print("resume 参数的不同格式")
    print("=" * 60)
    
    print("""
Command 的 resume 参数可以有多种格式：

1. 单个值：
   Command(resume="some value")
   - interrupt() 会直接返回这个值
   - 适用于简单的场景

2. 字典：
   Command(resume={"data": "some value"})
   - interrupt() 会返回整个字典
   - 适用于需要多个值的场景
   - 在这个例子中，我们使用这种格式

3. 中断 ID 到值的映射：
   Command(resume={interrupt_id: value})
   - 用于有多个中断的场景
   - 可以指定恢复哪个中断

示例代码：

# 在工具中：
def my_tool():
    result = interrupt({"query": "What is your name?"})
    # 恢复时，result 会是 Command.resume 的值
    return result["data"]  # 如果 resume={"data": "..."}

# 恢复执行：
command = Command(resume={"data": "John"})
graph.stream(command, config)
    """)


def explain_why_re_execute():
    """解释为什么节点会重新执行"""
    print("\n" + "=" * 60)
    print("为什么节点会重新执行（re-execute）？")
    print("=" * 60)
    
    print("""
关键点：节点会从开始重新执行，而不是从中断点继续

原因：
1. 简化实现
   - 不需要保存和恢复执行栈
   - 不需要处理部分执行的状态
   - 实现更简单、更可靠

2. 保证一致性
   - 节点的执行逻辑保持不变
   - 不需要特殊处理中断情况
   - 代码更容易理解和维护

3. 如何工作：
   ┌─────────────────────────────────────┐
   │ 第一次执行                          │
   └─────────────────────────────────────┘
   def human_assistance(query):
       print("开始执行")
       result = interrupt({"query": query})  ← 在这里中断
       print("这行不会执行")
       return result
   
   ┌─────────────────────────────────────┐
   │ 恢复执行（重新执行）                │
   └─────────────────────────────────────┘
   def human_assistance(query):
       print("开始执行")  ← 重新执行
       result = interrupt({"query": query})  ← 这次返回 resume 值
       print("这行会执行")  ← 继续执行
       return result

注意：
- 节点会完全重新执行
- 但 interrupt() 的行为会改变（返回 resume 值而不是抛异常）
- 这允许节点"假装"从未被中断
    """)


def explain_checkpointer_requirement():
    """解释为什么需要 checkpointer"""
    print("\n" + "=" * 60)
    print("为什么需要 checkpointer？")
    print("=" * 60)
    
    print("""
checkpointer 的作用：

1. 保存中断状态：
   - 保存图的状态（values）
   - 保存中断信息（interrupts）
   - 保存待执行的任务（tasks）
   - 保存下一个要执行的节点（next）

2. 恢复执行：
   - 从 checkpointer 读取保存的状态
   - 恢复中断信息
   - 知道从哪个节点继续执行

3. 没有 checkpointer 会怎样？
   - interrupt() 无法工作
   - 无法保存状态
   - 无法恢复执行
   - 会抛出错误

4. 如何启用 checkpointer：
   from langgraph.checkpoint.memory import MemorySaver
   
   memory = MemorySaver()
   graph = graph_builder.compile(checkpointer=memory)
   
   这样图就可以使用中断和恢复功能了
    """)


def demonstrate_flow():
    """演示完整的执行流程"""
    print("\n" + "=" * 60)
    print("完整执行流程演示")
    print("=" * 60)
    
    print("""
时间线：

T1: 用户发送消息
    "我需要一些关于构建 AI 代理的专家指导。"
    ↓
T2: chatbot 节点执行
    LLM 决定调用 human_assistance 工具
    返回包含 tool_calls 的消息
    ↓
T3: 路由到 tools 节点
    snapshot.next = ("tools",)
    ↓
T4: tools 节点开始执行
    ToolNode 调用 human_assistance 工具
    ↓
T5: human_assistance 工具执行
    print("interrupt 开始: ...")
    interrupt({"query": "..."}) 被调用
    ↓
T6: 抛出 GraphInterrupt 异常
    执行暂停
    状态保存到 checkpointer
    snapshot.next = ("chatbot",)
    ↓
T7: 生成器结束
    for 循环退出
    ↓
T8: 检查状态
    snapshot = graph.get_state(config)
    if snapshot.next:  # True
        # 图被中断了
    ↓
T9: 用户输入响应
    human_input = input("人工输入: ")
    ↓
T10: 创建 Command 对象
     human_command = Command(resume={"data": human_input})
     ↓
T11: 恢复执行
     events = graph.stream(human_command, config)
     ↓
T12: LangGraph 从 checkpointer 读取状态
     从 tools 节点重新开始执行
     ↓
T13: human_assistance 工具再次执行
     print("interrupt 开始: ...")
     interrupt({"query": "..."}) 再次被调用
     但这次检测到有 resume 值
     返回 {"data": human_input}
     ↓
T14: human_assistance 返回结果
     return human_response["data"]
     ↓
T15: tools 节点完成
     路由到 chatbot 节点
     ↓
T16: chatbot 节点处理工具结果
     生成最终响应
     ↓
T17: 图执行完成
     snapshot.next = ()
    """)


def main():
    """主函数"""
    explain_interrupt_command_mechanism()
    explain_resume_parameter()
    explain_why_re_execute()
    explain_checkpointer_requirement()
    demonstrate_flow()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
Command 对象如何恢复执行：

1. interrupt() 的双重行为：
   - 第一次：抛出异常，暂停执行
   - 恢复时：返回 resume 值，继续执行

2. Command 对象的作用：
   - 告诉 LangGraph 如何恢复执行
   - 指定 interrupt() 应该返回什么值

3. 执行流程：
   - 中断时：状态保存到 checkpointer
   - 恢复时：从 checkpointer 读取状态
   - 节点重新执行，但 interrupt() 返回 resume 值

4. 关键点：
   - 节点会完全重新执行（re-execute）
   - interrupt() 的行为取决于是否有 resume 值
   - 需要 checkpointer 来保存和恢复状态
   - Command(resume={...}) 的值会传递给 interrupt()

5. 实际应用：
   - 创建 Command 对象：Command(resume={"data": user_input})
   - 传入 graph.stream()：graph.stream(command, config)
   - LangGraph 会自动处理恢复逻辑
    """)


if __name__ == "__main__":
    main()

