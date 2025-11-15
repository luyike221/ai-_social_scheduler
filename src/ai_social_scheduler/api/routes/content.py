"""内容管理路由"""

from fastapi import APIRouter, Depends
from typing import List

from ...models.content import Content, ContentCreate
from ...repositories.content_repo import ContentRepository
from ..dependencies import get_content_repository

router = APIRouter()


@router.post("/", response_model=Content)
async def create_content(
    content: ContentCreate,
    repo: ContentRepository = Depends(get_content_repository),
):
    """创建内容"""
    data = content.model_dump()
    return await repo.create(data)


@router.get("/{content_id}", response_model=Content)
async def get_content(
    content_id: str,
    repo: ContentRepository = Depends(get_content_repository),
):
    """获取内容"""
    return await repo.get_by_id(content_id)


@router.get("/", response_model=List[Content])
async def list_contents(
    limit: int = 100,
    offset: int = 0,
    repo: ContentRepository = Depends(get_content_repository),
):
    """列出内容"""
    return await repo.list(limit=limit, offset=offset)

