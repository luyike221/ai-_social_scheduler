#!/bin/bash
# 部署脚本

set -e

echo "开始部署..."

# 安装依赖
poetry install

# 运行数据库迁移
poetry run python scripts/setup_db.py

# 运行测试
poetry run pytest

echo "部署完成"

