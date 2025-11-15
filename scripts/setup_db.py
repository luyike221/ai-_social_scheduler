"""数据库初始化脚本"""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.xiaohongshu_agent.core.config import settings


async def setup_database():
    """初始化数据库"""
    # TODO: 实现数据库初始化逻辑
    engine = create_async_engine(settings.database_url)
    
    # 创建表
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)
    
    print("数据库初始化完成")


if __name__ == "__main__":
    asyncio.run(setup_database())

