"""响应模型"""

from pydantic import BaseModel
from typing import Optional, Any, Dict


class SuccessResponse(BaseModel):
    """成功响应"""

    success: bool = True
    message: str = "操作成功"
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    """错误响应"""

    success: bool = False
    error: str
    detail: Optional[Dict[str, Any]] = None

