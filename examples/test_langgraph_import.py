"""
测试 LangGraph 快速入门示例的导入和基本功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """测试所有导入是否正常"""
    print("测试导入...")
    try:
        from ai_social_scheduler.study_langchain_or_langgraph.base.langgraph_quickstart import (
            example_1_create_agent,
            example_2_configure_llm,
            example_3_custom_prompt,
            example_3_dynamic_prompt,
            example_4_memory,
            example_5_structured_output,
            main,
        )
        print("✅ 所有导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_types():
    """验证所有 prompt 参数都是字符串类型"""
    print("\n验证 prompt 参数类型...")
    import inspect
    from ai_social_scheduler.study_langchain_or_langgraph.base import langgraph_quickstart
    
    functions = [
        langgraph_quickstart.example_1_create_agent,
        langgraph_quickstart.example_3_custom_prompt,
        langgraph_quickstart.example_3_dynamic_prompt,
    ]
    
    all_ok = True
    for func in functions:
        source = inspect.getsource(func)
        # 检查是否有 prompt= 参数
        if 'prompt=' in source:
            # 检查是否是列表
            if 'prompt=[' in source or 'prompt = [' in source:
                print(f"  ❌ {func.__name__}: prompt 是列表类型")
                all_ok = False
            else:
                print(f"  ✅ {func.__name__}: prompt 是字符串类型")
    
    return all_ok

if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 快速入门示例 - 导入测试")
    print("=" * 60)
    
    if test_imports():
        if test_prompt_types():
            print("\n✅ 所有检查通过！")
        else:
            print("\n❌ prompt 类型检查失败")
    else:
        print("\n❌ 导入测试失败")

