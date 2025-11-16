"""
LangGraph 快速入门示例运行脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ai_social_scheduler.study_langchain_or_langgraph.base.langgraph_quickstart import main
"""


if __name__ == "__main__":
    main()

