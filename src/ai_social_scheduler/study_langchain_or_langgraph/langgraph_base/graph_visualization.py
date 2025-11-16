"""
LangGraph 图可视化工具

本模块提供了可视化 LangGraph 图的功能，包括：
1. 生成 Mermaid 格式的图
2. 生成 ASCII 格式的图
3. 保存图为图片文件
4. 可配置的可视化参数

执行命令：
uv run python -m ai_social_scheduler.study_langchain_or_langgraph.langgraph_base.graph_visualization

"""

from pathlib import Path
from typing import Optional, Dict, Any
import os


# ==================== 可视化参数配置 ====================

class GraphVisualizationParams:
    """
    图可视化参数配置类
    
    用于配置 LangGraph 图的可视化选项
    """
    
    # Mermaid 图参数
    mermaid_theme: str = "default"  # Mermaid 主题：default, dark, forest, neutral
    mermaid_background: str = "white"  # 背景颜色
    mermaid_flowchart_direction: str = "TB"  # 流程图方向：TB (top-bottom), LR (left-right), BT (bottom-top), RL (right-left)
    mermaid_node_style: Dict[str, Any] = {
        "shape": "rounded",
        "fill": "#e1f5ff",
        "stroke": "#01579b",
        "strokeWidth": 2,
    }
    mermaid_edge_style: Dict[str, Any] = {
        "stroke": "#01579b",
        "strokeWidth": 2,
        "arrowhead": "vee",
    }
    
    # ASCII 图参数
    ascii_compact: bool = False  # 是否使用紧凑模式
    ascii_show_state: bool = True  # 是否显示状态信息
    
    # 图片保存参数
    image_format: str = "png"  # 图片格式：png, svg, pdf
    image_dpi: int = 300  # 图片分辨率（DPI）
    image_width: Optional[int] = None  # 图片宽度（像素），None 表示自动
    image_height: Optional[int] = None  # 图片高度（像素），None 表示自动
    image_background_color: str = "white"  # 图片背景颜色
    
    # 输出路径参数
    output_dir: str = "graph_visualizations"  # 输出目录
    save_mermaid_file: bool = True  # 是否保存 Mermaid 文件
    save_ascii_file: bool = True  # 是否保存 ASCII 文件
    save_image_file: bool = True  # 是否保存图片文件
    
    def __init__(
        self,
        mermaid_theme: Optional[str] = None,
        mermaid_background: Optional[str] = None,
        mermaid_flowchart_direction: Optional[str] = None,
        ascii_compact: Optional[bool] = None,
        ascii_show_state: Optional[bool] = None,
        image_format: Optional[str] = None,
        image_dpi: Optional[int] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ):
        """
        初始化可视化参数
        
        Args:
            mermaid_theme: Mermaid 主题
            mermaid_background: Mermaid 背景颜色
            mermaid_flowchart_direction: 流程图方向
            ascii_compact: ASCII 图是否紧凑
            ascii_show_state: ASCII 图是否显示状态
            image_format: 图片格式
            image_dpi: 图片分辨率
            output_dir: 输出目录
            **kwargs: 其他参数
        """
        if mermaid_theme is not None:
            self.mermaid_theme = mermaid_theme
        if mermaid_background is not None:
            self.mermaid_background = mermaid_background
        if mermaid_flowchart_direction is not None:
            self.mermaid_flowchart_direction = mermaid_flowchart_direction
        if ascii_compact is not None:
            self.ascii_compact = ascii_compact
        if ascii_show_state is not None:
            self.ascii_show_state = ascii_show_state
        if image_format is not None:
            self.image_format = image_format
        if image_dpi is not None:
            self.image_dpi = image_dpi
        if output_dir is not None:
            self.output_dir = output_dir
        
        # 更新其他参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ==================== 默认参数实例 ====================

# 默认的可视化参数
DEFAULT_VIS_PARAMS = GraphVisualizationParams()

# 暗色主题参数
DARK_THEME_PARAMS = GraphVisualizationParams(
    mermaid_theme="dark",
    mermaid_background="black",
    image_background_color="black",
)

# 紧凑模式参数
COMPACT_PARAMS = GraphVisualizationParams(
    ascii_compact=True,
    mermaid_flowchart_direction="LR",  # 横向布局更紧凑
)


# ==================== 可视化函数 ====================

def visualize_graph_mermaid(
    graph,
    params: Optional[GraphVisualizationParams] = None,
    output_path: Optional[str] = None
) -> str:
    """
    生成 Mermaid 格式的图可视化
    
    Args:
        graph: LangGraph 编译后的图对象
        params: 可视化参数，如果为 None 则使用默认参数
        output_path: 输出文件路径，如果为 None 则不保存文件
    
    Returns:
        str: Mermaid 格式的图代码
    """
    if params is None:
        params = DEFAULT_VIS_PARAMS
    
    try:
        # 获取图的 Mermaid 表示
        mermaid_code = graph.get_graph().draw_mermaid()
        
        # 如果指定了输出路径，保存文件
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(mermaid_code)
            print(f"✅ Mermaid 图已保存到: {output_file}")
        
        return mermaid_code
    except Exception as e:
        print(f"❌ 生成 Mermaid 图时出错: {e}")
        raise


def visualize_graph_ascii(
    graph,
    params: Optional[GraphVisualizationParams] = None,
    output_path: Optional[str] = None
) -> str:
    """
    生成 ASCII 格式的图可视化
    
    Args:
        graph: LangGraph 编译后的图对象
        params: 可视化参数，如果为 None 则使用默认参数
        output_path: 输出文件路径，如果为 None 则不保存文件
    
    Returns:
        str: ASCII 格式的图代码
    """
    if params is None:
        params = DEFAULT_VIS_PARAMS
    
    try:
        # 获取图的 ASCII 表示
        ascii_code = graph.get_graph().draw_ascii()
        
        # 如果指定了输出路径，保存文件
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(ascii_code)
            print(f"✅ ASCII 图已保存到: {output_file}")
        
        return ascii_code
    except Exception as e:
        print(f"❌ 生成 ASCII 图时出错: {e}")
        raise


def visualize_graph(
    graph,
    graph_name: str = "graph",
    params: Optional[GraphVisualizationParams] = None,
    show_mermaid: bool = True,
    show_ascii: bool = True,
    save_files: bool = True
) -> Dict[str, str]:
    """
    可视化 LangGraph 图（完整功能）
    
    Args:
        graph: LangGraph 编译后的图对象
        graph_name: 图的名称（用于生成文件名）
        params: 可视化参数，如果为 None 则使用默认参数
        show_mermaid: 是否显示 Mermaid 图
        show_ascii: 是否显示 ASCII 图
        save_files: 是否保存文件
    
    Returns:
        Dict[str, str]: 包含 'mermaid' 和 'ascii' 键的字典
    """
    if params is None:
        params = DEFAULT_VIS_PARAMS
    
    results = {}
    
    # 确保输出目录存在
    if save_files:
        output_dir = Path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成 Mermaid 图
    if show_mermaid:
        mermaid_output_path = None
        if save_files and params.save_mermaid_file:
            mermaid_output_path = output_dir / f"{graph_name}.mmd"
        
        mermaid_code = visualize_graph_mermaid(graph, params, mermaid_output_path)
        results["mermaid"] = mermaid_code
        
        if show_mermaid:
            print("\n" + "=" * 60)
            print("Mermaid 图代码:")
            print("=" * 60)
            print(mermaid_code)
            print("=" * 60)
    
    # 生成 ASCII 图
    if show_ascii:
        ascii_output_path = None
        if save_files and params.save_ascii_file:
            ascii_output_path = output_dir / f"{graph_name}.txt"
        
        ascii_code = visualize_graph_ascii(graph, params, ascii_output_path)
        results["ascii"] = ascii_code
        
        if show_ascii:
            print("\n" + "=" * 60)
            print("ASCII 图:")
            print("=" * 60)
            print(ascii_code)
            print("=" * 60)
    
    return results


# ==================== 示例使用 ====================

def example_usage():
    """
    示例：如何使用可视化功能
    """
    print("=" * 60)
    print("LangGraph 图可视化示例")
    print("=" * 60)
    print()
    
    # 导入 basic_chatbot 中的图创建函数
    from .basic_chatbot import create_chatbot_graph
    
    # 创建图
    print("创建聊天机器人图...")
    graph = create_chatbot_graph()
    print("✅ 图创建成功")
    print()
    
    # 使用默认参数可视化
    print("使用默认参数可视化图...")
    visualize_graph(
        graph,
        graph_name="basic_chatbot",
        show_mermaid=True,
        show_ascii=True,
        save_files=True
    )
    
    # 使用自定义参数可视化
    print("\n使用自定义参数可视化图...")
    custom_params = GraphVisualizationParams(
        mermaid_theme="dark",
        mermaid_flowchart_direction="LR",
        ascii_compact=True,
        output_dir="custom_visualizations"
    )
    visualize_graph(
        graph,
        graph_name="basic_chatbot_custom",
        params=custom_params,
        show_mermaid=True,
        show_ascii=True,
        save_files=True
    )


# ==================== 主函数 ====================

def main():
    """主函数"""
    try:
        example_usage()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

