"""依赖注入"""

from typing import Generator

from ..repositories.content_repo import ContentRepository
from ..repositories.interaction_repo import InteractionRepository
from ..repositories.analytics_repo import AnalyticsRepository
from ..services.llm_service import LLMService
from ..services.cache_service import CacheService
from ..services.storage_service import StorageService


def get_content_repository() -> Generator[ContentRepository, None, None]:
    """获取内容仓储"""
    repo = ContentRepository()
    try:
        yield repo
    finally:
        pass  # 清理资源


def get_interaction_repository() -> Generator[InteractionRepository, None, None]:
    """获取互动仓储"""
    repo = InteractionRepository()
    try:
        yield repo
    finally:
        pass


def get_analytics_repository() -> Generator[AnalyticsRepository, None, None]:
    """获取分析仓储"""
    repo = AnalyticsRepository()
    try:
        yield repo
    finally:
        pass


def get_llm_service() -> LLMService:
    """获取 LLM 服务"""
    return LLMService()


def get_cache_service() -> CacheService:
    """获取缓存服务"""
    return CacheService()


def get_storage_service() -> StorageService:
    """获取存储服务"""
    return StorageService()

