"""互动管理路由"""

from fastapi import APIRouter, Depends
from typing import List

from ...models.interaction import Interaction
from ...repositories.interaction_repo import InteractionRepository
from ..dependencies import get_interaction_repository

router = APIRouter()


@router.get("/", response_model=List[Interaction])
async def list_interactions(
    limit: int = 100,
    offset: int = 0,
    repo: InteractionRepository = Depends(get_interaction_repository),
):
    """列出互动"""
    return await repo.list(limit=limit, offset=offset)


@router.get("/{interaction_id}", response_model=Interaction)
async def get_interaction(
    interaction_id: str,
    repo: InteractionRepository = Depends(get_interaction_repository),
):
    """获取互动"""
    return await repo.get_by_id(interaction_id)

