"""Pytest 配置"""

import pytest
from typing import AsyncGenerator

from src.xiaohongshu_agent.core.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """测试配置"""
    return Settings(
        debug=True,
        database_url="sqlite+aiosqlite:///:memory:",
        redis_url="redis://localhost:6379/1",
    )


@pytest.fixture
async def mock_llm_service():
    """Mock LLM 服务"""
    # TODO: 实现 Mock LLM 服务
    yield None


@pytest.fixture
async def mock_cache_service():
    """Mock 缓存服务"""
    # TODO: 实现 Mock 缓存服务
    yield None

