"""LangGraph 基础示例模块"""

# 注意：文件已重新组织到子目录中
# 由于子目录名包含数字和中文，不能直接作为 Python 模块导入
# 请直接运行子目录中的文件，例如：
# uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.01-base.langgraph_quickstart
# 或者使用文件路径直接运行

# 如果需要导入，可以使用以下方式（但需要确保子目录名符合 Python 模块命名规范）：
try:
    # 尝试从新路径导入（如果子目录有 __init__.py 且命名规范）
    from . import basic_chatbot  # noqa: F401
except ImportError:
    # 如果导入失败，说明文件已移动到子目录
    # 请直接运行子目录中的文件
    pass

__all__ = [
    # 由于文件已移动到子目录，不再导出这些函数
    # 请直接运行子目录中的文件
]

