"""数据分析路由"""

from fastapi import APIRouter, Depends
from datetime import datetime
from typing import List

from ...models.analytics import AnalyticsMetrics, AnalyticsReport
from ...repositories.analytics_repo import AnalyticsRepository
from ..dependencies import get_analytics_repository

router = APIRouter()


@router.get("/metrics", response_model=List[AnalyticsMetrics])
async def list_metrics(
    limit: int = 100,
    offset: int = 0,
    repo: AnalyticsRepository = Depends(get_analytics_repository),
):
    """列出分析指标"""
    return await repo.list(limit=limit, offset=offset)


@router.get("/report", response_model=AnalyticsReport)
async def get_report(
    start_date: datetime,
    end_date: datetime,
    repo: AnalyticsRepository = Depends(get_analytics_repository),
):
    """获取分析报告"""
    return await repo.get_report(start_date, end_date)

