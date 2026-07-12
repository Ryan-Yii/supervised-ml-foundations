"""第一周脚本共享的路径和保存工具。"""

from __future__ import annotations

from pathlib import Path


OUTPUTS_ROOT = Path("outputs")


def ensure_output_dir(name: str) -> Path:
    """创建并返回 ``outputs/<name>`` 相对目录。"""
    directory = OUTPUTS_ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def output_file(directory_name: str, filename: str) -> Path:
    """创建输出目录并返回其中一个相对文件路径。"""
    return ensure_output_dir(directory_name) / filename


def configure_chinese_plot_font() -> str:
    """配置一个本机可用的中文 Matplotlib 字体并返回其名称。"""
    from matplotlib import font_manager, rcParams

    preferred_fonts = (
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft JhengHei",
    )
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((font for font in preferred_fonts if font in available_fonts), None)
    if selected_font is None:
        raise RuntimeError(
            "未找到可用的中文绘图字体。请安装 Microsoft YaHei、Noto Sans CJK SC 或 SimHei。"
        )
    rcParams["font.sans-serif"] = [selected_font]
    rcParams["axes.unicode_minus"] = False
    return selected_font
