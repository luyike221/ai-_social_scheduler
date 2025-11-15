"""工作流控制路由"""

from fastapi import APIRouter
from typing import Dict, Any

from ...graph.workflow import run_workflow

router = APIRouter()


@router.post("/run")
async def run_workflow_endpoint(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    """运行工作流"""
    result = await run_workflow(initial_state)
    return result

