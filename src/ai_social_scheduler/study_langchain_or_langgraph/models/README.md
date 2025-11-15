# 阿里百炼模型测试

## 说明

此目录包含使用 LangChain 的 OpenAI 接口连接阿里百炼模型的测试代码。

## 前置条件

1. 确保已安装项目依赖：
   ```bash
   uv sync
   ```

2. 配置环境变量（在 `.env` 文件中）：
   ```env
   ALIBABA_BAILIAN_API_KEY=your_api_key_here
   ALIBABA_BAILIAN_API_SECRET=your_api_secret_here
   ALIBABA_BAILIAN_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1
   ALIBABA_BAILIAN_MODEL=qwen-plus
   ```

## 运行测试

```bash
# 使用 uv 运行
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.models.test_alibaba_bailian

# 或者直接运行
uv run python src/ai_social_scheduler/study_langchain_or_langgraph/models/test_alibaba_bailian.py
```

## 测试内容

1. **基本连接测试**：测试使用 LangChain 的 ChatOpenAI 接口连接阿里百炼
2. **流式响应测试**：测试流式输出功能

## 注意事项

- 阿里百炼提供了 OpenAI 兼容模式的 API，可以直接使用 LangChain 的 ChatOpenAI
- 如果 API Key 不是以 "sk-" 开头，代码会自动添加前缀
- 如果连接失败，请检查：
  - API Key 和 API Secret 是否正确
  - 网络连接是否正常
  - 端点 URL 是否正确

