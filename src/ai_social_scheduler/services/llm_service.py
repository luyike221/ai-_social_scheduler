"""LLM 调用服务"""

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage

from ..config import settings
from ..core.exceptions import LLMError
from ..tools.logging import get_logger

logger = get_logger(__name__)


class LLMService:
    """LLM 服务"""

    def __init__(self):
        self.provider = settings.default_llm_provider
        self.model = settings.default_model
        self._client = None

    def _get_client(self):
        """获取 LLM 客户端"""
        if self._client is None:
            if self.provider == "openai":
                self._client = ChatOpenAI(
                    model=self.model,
                    openai_api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                )
            elif self.provider == "anthropic":
                self._client = ChatAnthropic(
                    model=self.model,
                    anthropic_api_key=settings.anthropic_api_key,
                )
            else:
                raise LLMError(f"不支持的 LLM 提供商: {self.provider}")
        return self._client

    async def generate(self, messages: List[BaseMessage], **kwargs) -> str:
        """生成文本"""
        try:
            client = self._get_client()
            response = await client.ainvoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error("LLM 调用失败", error=str(e))
            raise LLMError(f"LLM 调用失败: {str(e)}") from e

    async def generate_stream(self, messages: List[BaseMessage], **kwargs):
        """流式生成文本"""
        try:
            client = self._get_client()
            async for chunk in client.astream(messages, **kwargs):
                yield chunk.content
        except Exception as e:
            logger.error("LLM 流式调用失败", error=str(e))
            raise LLMError(f"LLM 流式调用失败: {str(e)}") from e

