"""请求模型"""

from pydantic import BaseModel
from typing import Optional, Dict, Any


class WorkflowRunRequest(BaseModel):
    """工作流运行请求"""

    workflow_id: str
    initial_state: Dict[str, Any]


class ContentSearchRequest(BaseModel):
    """内容搜索请求"""

    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    offset: int = 0

