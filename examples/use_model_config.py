"""模型配置使用示例"""

from ai_social_scheduler.config import model_config


def example_alibaba_bailian():
    """使用阿里百炼模型配置示例"""
    try:
        # 获取阿里百炼配置
        bailian_config = model_config.get_alibaba_bailian_config()
        
        print("阿里百炼配置:")
        print(f"  API Key: {bailian_config.api_key[:10]}...")
        print(f"  Endpoint: {bailian_config.endpoint}")
        print(f"  Model: {bailian_config.model}")
        print(f"  Temperature: {bailian_config.temperature}")
        print(f"  Timeout: {bailian_config.timeout}秒")
        
        # 使用配置调用 API
        # import httpx
        # response = httpx.post(
        #     bailian_config.endpoint,
        #     headers={"Authorization": f"Bearer {bailian_config.api_key}"},
        #     json={"model": bailian_config.model, "messages": [...]},
        #     timeout=bailian_config.timeout,
        # )
        
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请确保在 .env 文件中配置了 ALIBABA_BAILIAN_API_KEY 和 ALIBABA_BAILIAN_API_SECRET")


def example_deepseek_ocr():
    """使用 DeepSeek OCR 模型配置示例"""
    try:
        # 获取 DeepSeek OCR 配置
        ocr_config = model_config.get_deepseek_ocr_config()
        
        print("\nDeepSeek OCR 配置:")
        print(f"  API Key: {ocr_config.api_key[:10]}...")
        print(f"  Endpoint: {ocr_config.endpoint}")
        print(f"  Model: {ocr_config.model}")
        print(f"  Timeout: {ocr_config.timeout}秒")
        print(f"  Max Retries: {ocr_config.max_retries}")
        
        # 使用配置调用 API
        # import httpx
        # response = httpx.post(
        #     ocr_config.endpoint,
        #     headers={"Authorization": f"Bearer {ocr_config.api_key}"},
        #     json={"model": ocr_config.model, "messages": [...]},
        #     timeout=ocr_config.timeout,
        # )
        
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请确保在 .env 文件中配置了 DEEPSEEK_OCR_API_KEY")


if __name__ == "__main__":
    example_alibaba_bailian()
    example_deepseek_ocr()

