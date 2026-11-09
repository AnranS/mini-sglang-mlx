"""
ch10_profiling.py 的画图工具：把分析结果画成 PNG（同时输出 light / dark 两个版本，供文档用 <picture> 切换）。

设计约定（与 docs 中所有图一致）：
  * 1px = 1/100 inch，按 200 dpi 输出（高分屏 2x）
  * 柱子最粗 24px，数据端 4px 圆角、基线端是直角；堆叠段之间留 2px 背景色间隙
  * 折线 2px，端点圆点 8px；网格线/坐标轴是 1px 实线，颜色比背景深一档
  * 分类颜色按固定顺序取（已用 validate_palette 校验 light / dark 两套）
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

PX = 1 / 100  # 1px 对应的英寸
PT = 72 * PX  # 1px 对应的磅

THEMES: Dict[str, Dict] = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    ),
    "dark": dict(
        surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
        series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
    ),
}

import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 中文字体缺少某个字重时的警告
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def new_figure(theme: Dict, width_px: int, height_px: int, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(width_px * PX, height_px * PX), squeeze=False)
    fig.patch.set_facecolor(theme["surface"])
    for ax in axes[0]:
        style_axes(ax, theme)
    return fig, list(axes[0])


def style_axes(ax, theme: Dict, grid_axis: str = "x") -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.spines["bottom"].set_linewidth(1 * PT)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0, pad=6)
    ax.grid(False)
    ax.grid(axis=grid_axis, color=theme["grid"], linewidth=1 * PT, linestyle="-")
    ax.set_axisbelow(True)


def title(fig, text: str, subtitle: str, theme: Dict) -> None:
    fig.text(0.012, 0.965, text, ha="left", va="top", fontsize=12, color=theme["ink"], weight="bold")
    fig.text(0.012, 0.965 - 22 / (fig.get_figheight() / PX), subtitle, ha="left", va="top", fontsize=9,
             color=theme["ink2"])


def _px_to_data(ax, px_x: float, px_y: float):
    """把像素长度换算成数据坐标长度（用于画固定像素大小的圆角、柱宽）。"""
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())  # inches
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = px_x * PX / bbox.width * (x1 - x0)
    dy = px_y * PX / bbox.height * abs(y1 - y0)
    return dx, dy


def hbar(ax, y: float, left: float, width: float, color: str, theme: Dict, height_px: float = 20,
         round_end: bool = True, gap: bool = False) -> None:
    """一段横向柱子：基线端直角，数据端 4px 圆角；gap=True 时右侧留 2px 背景色间隙（堆叠用）。"""
    if width <= 0:
        return
    rx, h = _px_to_data(ax, 4, height_px)
    gap_x = _px_to_data(ax, 2, 0)[0] if gap else 0.0
    w = max(width - gap_x, 0.0)
    if not round_end or w <= 2 * rx:
        ax.add_patch(Rectangle((left, y - h / 2), w, h, facecolor=color, edgecolor="none", zorder=3))
        return
    ry = _px_to_data(ax, 0, 4)[1]
    ax.add_patch(Rectangle((left, y - h / 2), w - rx, h, facecolor=color, edgecolor="none", zorder=3))
    # 末端的小圆角矩形：mutation_aspect 修正 x/y 数据单位不同导致的圆角变形（让圆角在像素上是正圆）
    ax.add_patch(FancyBboxPatch((left + w - 2 * rx, y - h / 2), 2 * rx, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                                mutation_aspect=ry / rx, facecolor=color, edgecolor="none", zorder=3))


def legend_row(fig, items: Sequence[tuple], theme: Dict, y: float) -> None:
    """图例：一行色块 + 名称，放在标题下方（文字用正文颜色，不用系列颜色）。"""
    x = 0.012
    for name, color in items:
        fig.patches.append(Rectangle((x, y - 0.012), 10 * PX / fig.get_figwidth(), 10 * PX / fig.get_figheight(),
                                     transform=fig.transFigure, facecolor=color, edgecolor="none"))
        t = fig.text(x + 16 * PX / fig.get_figwidth(), y - 0.012 + 5 * PX / fig.get_figheight(), name,
                     fontsize=9, color=theme["ink2"], va="center")
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(fig.transFigure.inverted())
        x = bb.x1 + 18 * PX / fig.get_figwidth()


def save_both(draw: Callable[[Dict], "plt.Figure"], outdir: str, stem: str) -> List[str]:
    """用两套主题各画一次，保存为 <stem>-light.png / <stem>-dark.png。"""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for mode, theme in THEMES.items():
        fig = draw(theme)
        path = os.path.join(outdir, f"{stem}-{mode}.png")
        fig.savefig(path, dpi=200, facecolor=theme["surface"])
        plt.close(fig)
        paths.append(path)
    return paths
