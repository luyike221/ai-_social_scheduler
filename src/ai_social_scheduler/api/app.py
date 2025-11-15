"""FastAPI 应用实例"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import settings
from ..tools.logging import configure_logging
from .routes import content, interaction, analytics, workflow

# 配置日志
configure_logging()

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(content.router, prefix="/api/v1/content", tags=["content"])
app.include_router(interaction.router, prefix="/api/v1/interaction", tags=["interaction"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(workflow.router, prefix="/api/v1/workflow", tags=["workflow"])


@app.get("/")
async def root():
    """根路径"""
    return {"message": "小红书运营 Agent API", "version": settings.app_version}


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

