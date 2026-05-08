"""
미적분 탐험대 — 발표용 PPT 자동 생성 스크립트
python-pptx 사용
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

# ── 한글 폰트 ────────────────────────────────────────────────────────────────
def _set_korean_font():
    # 캐시 삭제 후 재스캔하여 새로 설치된 폰트 인식
    font_manager._load_fontmanager(try_read_cache=False)
    candidates = [
        "NanumGothic", "NanumBarunGothic", "NanumMyeongjo",
        "Malgun Gothic", "AppleGothic", "Gulim",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    # 폴백: ttf 파일 직접 지정
    import glob
    nanum_paths = glob.glob("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
    if nanum_paths:
        font_manager.fontManager.addfont(nanum_paths[0])
        prop = font_manager.FontProperties(fname=nanum_paths[0])
        plt.rcParams["font.family"] = prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False
        return prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"

FONT_NAME = _set_korean_font()
plt.rcParams["axes.unicode_minus"] = False

# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
BLUE   = RGBColor(0x19, 0x76, 0xD2)
DBLUE  = RGBColor(0x1A, 0x23, 0x7E)
GREEN  = RGBColor(0x4C, 0xAF, 0x50)
ORANGE = RGBColor(0xFF, 0x98, 0x00)
RED    = RGBColor(0xF4, 0x43, 0x36)
PURPLE = RGBColor(0x9C, 0x27, 0xB0)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
GRAY   = RGBColor(0xF5, 0xF5, 0xF5)
DARK   = RGBColor(0x21, 0x21, 0x21)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def add_rect(slide, x, y, w, h, fill_rgb, alpha=None):
    shape = slide.shapes.add_shape(
        1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.line.fill.background()
    fill = shape.fill
    fill.solid()
    fill.fore_color.rgb = fill_rgb
    return shape


def add_text(slide, text, x, y, w, h,
             font_size=24, bold=False, color=DARK, align=PP_ALIGN.LEFT,
             wrap=True):
    txBox = slide.shapes.add_textbox(
        Inches(x), Inches(y), Inches(w), Inches(h))
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Malgun Gothic"
    return txBox


def fig_to_stream(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    return buf


def add_figure(slide, fig, x, y, w, h):
    buf = fig_to_stream(fig)
    slide.shapes.add_picture(buf, Inches(x), Inches(y), Inches(w), Inches(h))
    plt.close(fig)


def slide_bg(slide, color: RGBColor):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


# ── 그래프 생성 ───────────────────────────────────────────────────────────────
def fig_number_line():
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.set_xlim(-6, 6); ax.set_ylim(-0.8, 0.8)
    ax.axhline(0, color="#333", linewidth=2.5)
    for x in range(-5, 6):
        ax.plot(x, 0, "o", color="#555", markersize=7)
        ax.text(x, -0.4, str(x), ha="center", va="top", fontsize=12)
    ax.annotate("", xy=(6, 0), xytext=(5.4, 0),
                arrowprops=dict(arrowstyle="->", color="#333", lw=2))
    ax.annotate("", xy=(-6, 0), xytext=(-5.4, 0),
                arrowprops=dict(arrowstyle="->", color="#333", lw=2))
    ax.plot(3, 0, "o", color="#F44336", markersize=14, zorder=5)
    ax.text(3, 0.4, "숫자 3의 위치!", ha="center", fontsize=12,
            color="#F44336", fontweight="bold")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig


def fig_slope():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.linspace(-1, 5, 200)
    ax.plot(x, 1.5 * x + 0.5, color="#1976D2", lw=2.5, label="기울기 = 1.5")
    ax.plot([1, 3], [2.0, 5.0], "ro", markersize=9)
    ax.annotate("", xy=(3, 2.0), xytext=(1, 2.0),
                arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=2))
    ax.annotate("", xy=(3, 5.0), xytext=(3, 2.0),
                arrowprops=dict(arrowstyle="->", color="#F44336", lw=2))
    ax.text(2, 1.5, "가로 +2", ha="center", color="#4CAF50", fontsize=11)
    ax.text(3.4, 3.5, "세로 +3", ha="left", color="#F44336", fontsize=11)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color="#888", lw=0.8)
    ax.set_title("기울기 = 3 ÷ 2 = 1.5", fontsize=13)
    fig.patch.set_facecolor("white")
    return fig


def fig_quadratic():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.linspace(-3.5, 3.5, 300)
    ax.plot(x, x**2, color="#9C27B0", lw=2.8, label="y = x²")
    ax.plot(x, -0.5*x**2 + 4, color="#FF9800", lw=2.5, linestyle="--",
            label="y = -0.5x² + 4")
    ax.plot(0, 0, "y*", markersize=15, zorder=5)
    ax.plot(0, 4, "g*", markersize=15, zorder=5)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color="#888", lw=0.8)
    ax.set_ylim(-2, 9); ax.set_title("이차함수 그래프", fontsize=13)
    fig.patch.set_facecolor("white")
    return fig


def fig_derivative():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.linspace(-0.3, 3.5, 300)
    x0 = 2.0
    ax1.plot(x, x**2, color="#1976D2", lw=2.5, label="f(x) = x²")
    colors = ["#FFCDD2", "#EF9A9A", "#E57373", "#F44336", "#B71C1C"]
    for i, h in enumerate([1.5, 0.8, 0.4, 0.15, 0.03]):
        m = ((x0 + h)**2 - x0**2) / h
        b = x0**2 - m * x0
        xs = np.linspace(x0 - 0.3, x0 + h + 0.1, 30)
        ax1.plot(xs, m * xs + b, color=colors[i], lw=1.6,
                 label=f"h={h:.2f}")
    ax1.set_title("h → 0 으로 수렴!", fontsize=12)
    ax1.legend(fontsize=7.5, loc="upper left"); ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.3, 3.5); ax1.set_ylim(-0.5, 10)

    ax2.plot(x, x**2, color="#1976D2", lw=2.5)
    m_t = 2 * x0; b_t = x0**2 - m_t * x0
    xs_t = np.linspace(x0 - 1.5, x0 + 1.5, 60)
    ax2.plot(xs_t, m_t * xs_t + b_t, color="#F44336", lw=2.8,
             label=f"접선 기울기 = f'({x0}) = {m_t}")
    ax2.plot(x0, x0**2, "ko", markersize=8)
    ax2.set_title(f"f'({x0}) = {m_t} (미분값)", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.3, 3.5); ax2.set_ylim(-0.5, 10)
    plt.tight_layout()
    fig.patch.set_facecolor("white")
    return fig


def fig_riemann(n=20):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fn = lambda x: x**2
    for ax, ni, title in zip(axes, [4, 10, n],
                              ["n=4 (거칠게)", "n=10 (보통)", f"n={n} (정밀하게)"]):
        x = np.linspace(-0.2, 3.2, 300)
        ax.plot(x, [fn(xi) for xi in x], color="#1976D2", lw=2.5, zorder=4)
        dx = 3.0 / ni
        area = 0
        for xi in np.linspace(0, 3 - dx, ni):
            h = fn(xi + dx / 2)
            area += h * dx
            ax.add_patch(mpatches.Rectangle(
                (xi, 0), dx, h, lw=0.5, ec="#333", fc="#4CAF50", alpha=0.45))
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xlim(-0.2, 3.2); ax.set_ylim(0, 10)
        ax.set_title(f"{title}\n≈ {area:.3f}", fontsize=11)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.patch.set_facecolor("white")
    return fig


def fig_integral():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fn = lambda x: x**2
    x_all = np.linspace(-0.2, 3.2, 300)
    x_fill = np.linspace(0, 3, 300)
    ax.plot(x_all, [fn(xi) for xi in x_all], color="#1976D2", lw=2.5,
            label="f(x) = x²")
    ax.fill_between(x_fill, 0, [fn(xi) for xi in x_fill],
                    color="#FF9800", alpha=0.55, label="넓이 = 9 (= 3³/3)")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("∫₀³ x² dx = 9", fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.25)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.patch.set_facecolor("white")
    return fig


def fig_ai_tutor():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    boxes = [
        (0.05, 0.6, "초등학생 질문\n'미분이 뭐야?'", "#E3F2FD", "#1976D2"),
        (0.38, 0.6, "Ollama\ngemma4 모델", "#F3E5F5", "#9C27B0"),
        (0.70, 0.6, "친절한 설명\n비유와 예시로!", "#E8F5E9", "#4CAF50"),
        (0.38, 0.15, "Streamlit UI\n대화형 채팅", "#FFF9C4", "#FF9800"),
    ]
    for (bx, by, txt, bg, fc) in boxes:
        fancy = mpatches.FancyBboxPatch(
            (bx, by), 0.22, 0.28,
            boxstyle="round,pad=0.02",
            facecolor=bg, edgecolor=fc, linewidth=2)
        ax.add_patch(fancy)
        ax.text(bx + 0.11, by + 0.14, txt, ha="center", va="center",
                fontsize=11, color=fc, fontweight="bold",
                multialignment="center")
    # 화살표
    for (x1, x2, y) in [(0.27, 0.38, 0.74), (0.60, 0.70, 0.74)]:
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("AI 튜터 시스템 구조", fontsize=13, pad=10)
    fig.patch.set_facecolor("white")
    return fig


# ── 슬라이드 레이아웃 함수들 ─────────────────────────────────────────────────

def make_title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    slide_bg(slide, DBLUE)
    # 배경 장식 원
    for (x, y, w, h, c) in [
        (10.5, -0.5, 3.5, 3.5, RGBColor(0x28, 0x3A, 0x8E)),
        (11.8, 5.5, 2.5, 2.5, RGBColor(0x28, 0x3A, 0x8E)),
        (-0.5, 4.5, 2.5, 2.5, RGBColor(0x28, 0x3A, 0x8E)),
    ]:
        s = slide.shapes.add_shape(9, Inches(x), Inches(y), Inches(w), Inches(h))
        s.fill.solid(); s.fill.fore_color.rgb = c
        s.line.fill.background()

    add_text(slide, "🚀 미적분 탐험대",
             0.8, 1.5, 11.5, 1.8, font_size=54, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)
    add_text(slide, "초등학생도 이해하는 미분 · 적분 여행",
             1.0, 3.3, 11.0, 0.9, font_size=28, color=RGBColor(0xBB, 0xDE, 0xFB),
             align=PP_ALIGN.CENTER)
    add_text(slide, "Streamlit + Ollama (gemma4) 기반 스토리텔링 학습",
             1.5, 4.2, 10.0, 0.7, font_size=18, color=RGBColor(0x90, 0xCA, 0xF9),
             align=PP_ALIGN.CENTER)
    add_text(slide, "📏 수직선  →  📊 함수  →  📐 기울기  →  🎢 이차함수  →  ⚡ 미분  →  🏞️ 적분",
             0.8, 5.2, 11.5, 0.7, font_size=16, color=RGBColor(0xE3, 0xF2, 0xFD),
             align=PP_ALIGN.CENTER)
    return slide


def make_overview_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide_bg(slide, GRAY)
    add_rect(slide, 0, 0, 13.33, 1.1, BLUE)
    add_text(slide, "📚 학습 여정 — 6단계 로드맵",
             0.3, 0.1, 12.5, 0.9, font_size=30, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)

    steps = [
        ("1", "📏 수직선", "숫자들의 집\n양수·음수·0"),
        ("2", "📊 함숫값", "마법 상자 함수\n입력 → 출력"),
        ("3", "📐 기울기", "얼마나 가파를까?\n세로÷가로"),
        ("4", "🎢 이차함수", "포물선의 세계\ny=ax²+bx+c"),
        ("5", "⚡ 미분", "순간 기울기!\n극한 → 접선"),
        ("6", "🏞️ 적분", "넓이를 구하자!\n무한 직사각형"),
    ]
    colors = [BLUE, GREEN, ORANGE, PURPLE, RED, RGBColor(0x00, 0x96, 0x88)]
    for i, (num, title, sub) in enumerate(steps):
        col = i % 3
        row = i // 3
        bx = 0.4 + col * 4.2
        by = 1.4 + row * 2.6
        s = slide.shapes.add_shape(
            1, Inches(bx), Inches(by), Inches(3.7), Inches(2.2))
        s.fill.solid(); s.fill.fore_color.rgb = colors[i]
        s.line.fill.background()
        add_text(slide, f"{num}", bx + 0.1, by + 0.1, 0.5, 0.5,
                 font_size=22, bold=True, color=WHITE)
        add_text(slide, title, bx + 0.15, by + 0.55, 3.3, 0.65,
                 font_size=20, bold=True, color=WHITE)
        add_text(slide, sub, bx + 0.15, by + 1.15, 3.3, 0.85,
                 font_size=13, color=RGBColor(0xFF, 0xFF, 0xFF))
    return slide


def make_concept_slide(prs, chapter_num, title, story, concepts, fig=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide_bg(slide, WHITE)

    # 헤더
    add_rect(slide, 0, 0, 13.33, 1.1,
             [BLUE, GREEN, ORANGE, PURPLE, RED, RGBColor(0x00, 0x96, 0x88)][chapter_num - 1])
    add_text(slide, f"Chapter {chapter_num}. {title}",
             0.3, 0.1, 12.5, 0.9, font_size=30, bold=True, color=WHITE,
             align=PP_ALIGN.LEFT)

    # 스토리 박스
    add_rect(slide, 0.3, 1.25, 5.8 if fig else 12.7, 1.7,
             RGBColor(0xE8, 0xF4, 0xFD))
    add_text(slide, f"💬 {story}",
             0.45, 1.3, 5.5 if fig else 12.4, 1.6, font_size=14,
             color=RGBColor(0x0D, 0x47, 0xA1))

    # 개념 목록
    y_start = 3.1
    for bullet in concepts:
        add_text(slide, f"• {bullet}",
                 0.4, y_start, 5.6 if fig else 12.5, 0.5,
                 font_size=15, color=DARK)
        y_start += 0.52

    # 그래프
    if fig:
        add_figure(slide, fig, 6.4, 1.2, 6.6, 5.8)

    return slide


def make_derivative_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide_bg(slide, WHITE)
    add_rect(slide, 0, 0, 13.33, 1.1, RED)
    add_text(slide, "⚡ Chapter 5. 미분은 기울기다",
             0.3, 0.1, 12.5, 0.9, font_size=30, bold=True, color=WHITE)

    # 공식 강조
    add_rect(slide, 0.3, 1.2, 5.4, 1.8, RGBColor(0xFF, 0xF9, 0xC4))
    add_text(slide, "핵심 공식",
             0.5, 1.25, 4.8, 0.45, font_size=16, bold=True,
             color=RGBColor(0xE6, 0x51, 0x00))
    add_text(slide, "f'(x) = lim  [f(x+h) - f(x)] / h\n           h→0",
             0.5, 1.7, 5.0, 0.95, font_size=18, bold=True,
             color=DARK)

    bullets = [
        "h를 점점 0에 가깝게 → 극한값 = 미분값",
        "미분값 = 그 점에서의 접선 기울기",
        "f(x)=x² → f'(x)=2x  (n·xⁿ⁻¹ 공식)",
        "양의 미분값 → 증가 / 음의 미분값 → 감소",
        "미분값=0 → 극대 또는 극소 (꼭짓점!)",
    ]
    y_s = 3.15
    for b in bullets:
        add_text(slide, f"• {b}", 0.4, y_s, 5.6, 0.5, font_size=14, color=DARK)
        y_s += 0.52

    fig = fig_derivative()
    add_figure(slide, fig, 6.1, 1.15, 6.9, 5.9)
    return slide


def make_integral_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide_bg(slide, WHITE)
    add_rect(slide, 0, 0, 13.33, 1.1, RGBColor(0x00, 0x96, 0x88))
    add_text(slide, "🏞️ Chapter 6. 적분은 넓이다",
             0.3, 0.1, 12.5, 0.9, font_size=30, bold=True, color=WHITE)

    add_rect(slide, 0.3, 1.2, 5.4, 1.8, RGBColor(0xE8, 0xF5, 0xE9))
    add_text(slide, "핵심 공식",
             0.5, 1.25, 4.8, 0.45, font_size=16, bold=True, color=GREEN)
    add_text(slide, "∫ₐᵇ f(x) dx = F(b) - F(a)",
             0.5, 1.7, 5.0, 0.85, font_size=20, bold=True, color=DARK)

    bullets = [
        "곡선 아래를 얇은 직사각형으로 채움",
        "직사각형 넓이 = f(x) × Δx",
        "Δx → 0 (무한히 얇게) → 정확한 넓이",
        "리만 합 → 정적분으로 수렴",
        "미적분학 기본 정리: (미분의 역=적분)",
    ]
    y_s = 3.15
    for b in bullets:
        add_text(slide, f"• {b}", 0.4, y_s, 5.6, 0.5, font_size=14, color=DARK)
        y_s += 0.52

    fig = fig_riemann(20)
    add_figure(slide, fig, 6.0, 1.2, 7.0, 3.0)
    fig2 = fig_integral()
    add_figure(slide, fig2, 6.0, 4.2, 7.0, 3.2)
    return slide


def make_ai_tutor_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide_bg(slide, RGBColor(0xF3, 0xE5, 0xF5))
    add_rect(slide, 0, 0, 13.33, 1.1, PURPLE)
    add_text(slide, "🤖 AI 수학 튜터 — Ollama + gemma4",
             0.3, 0.1, 12.5, 0.9, font_size=30, bold=True, color=WHITE)

    add_text(slide, "시스템 구성",
             0.4, 1.3, 5.5, 0.55, font_size=20, bold=True, color=PURPLE)

    tech_items = [
        ("🖥️ Streamlit", "대화형 웹 UI, 실시간 렌더링"),
        ("🤖 Ollama", "로컬 LLM 실행 (gemma4:e4b / 26b)"),
        ("📊 Matplotlib", "수학 그래프 시각화"),
        ("🔢 NumPy", "수치 계산 및 배열 처리"),
        ("📐 SymPy", "기호 수학 (미분·적분 공식)"),
    ]
    y_t = 1.95
    for icon_name, desc in tech_items:
        add_rect(slide, 0.4, y_t, 5.3, 0.52, WHITE)
        add_text(slide, f"{icon_name}  —  {desc}",
                 0.55, y_t + 0.05, 5.0, 0.42, font_size=13, color=DARK)
        y_t += 0.58

    add_text(slide, "모델 비교",
             0.4, 5.05, 5.5, 0.5, font_size=18, bold=True, color=PURPLE)
    add_rect(slide, 0.4, 5.55, 2.5, 1.0, RGBColor(0xE8, 0xF4, 0xFD))
    add_text(slide, "gemma4:e4b\n⚡ 빠른 응답\n가벼운 질문에 최적",
             0.5, 5.6, 2.3, 0.9, font_size=12, color=BLUE)
    add_rect(slide, 3.1, 5.55, 2.5, 1.0, RGBColor(0xFB, 0xE9, 0xE7))
    add_text(slide, "gemma4:26b\n🧠 깊은 설명\n복잡한 개념 풀이",
             3.2, 5.6, 2.3, 0.9, font_size=12, color=RED)

    fig = fig_ai_tutor()
    add_figure(slide, fig, 6.3, 1.2, 6.7, 5.0)
    return slide


def make_summary_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide_bg(slide, DBLUE)
    add_text(slide, "🎓 핵심 요약",
             0.5, 0.4, 12.0, 1.0, font_size=38, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)

    summaries = [
        ("📏 수직선", "모든 수를 한 줄에 — 수학의 기초"),
        ("📊 함수", "입력 → 규칙 → 출력 (마법 상자)"),
        ("📐 기울기", "세로 변화 ÷ 가로 변화 = 가파름"),
        ("🎢 이차함수", "y=ax²+bx+c, 포물선, 꼭짓점"),
        ("⚡ 미분", "극한으로 구한 순간 기울기 = f'(x)"),
        ("🏞️ 적분", "무한 직사각형의 합 = 곡선 아래 넓이"),
    ]
    colors2 = [BLUE, GREEN, ORANGE, PURPLE, RED, RGBColor(0x00, 0x96, 0x88)]
    for i, (title_s, desc_s) in enumerate(summaries):
        col = i % 2
        row = i // 2
        bx = 0.5 + col * 6.4
        by = 1.7 + row * 1.8
        add_rect(slide, bx, by, 5.9, 1.4, colors2[i])
        add_text(slide, title_s, bx + 0.15, by + 0.1, 5.5, 0.5,
                 font_size=18, bold=True, color=WHITE)
        add_text(slide, desc_s, bx + 0.15, by + 0.65, 5.5, 0.6,
                 font_size=13, color=WHITE)

    add_text(slide, "🤖 + Streamlit + Ollama(gemma4) = 누구나 미적분을 배울 수 있다!",
             0.5, 7.0, 12.2, 0.45, font_size=16, bold=True,
             color=RGBColor(0xBB, 0xDE, 0xFB), align=PP_ALIGN.CENTER)
    return slide


# ── 메인 ─────────────────────────────────────────────────────────────────────

def build_ppt(output_path="미적분_탐험대_발표자료.pptx"):
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    # 1. 표지
    make_title_slide(prs)

    # 2. 로드맵
    make_overview_slide(prs)

    # 3. 수직선
    make_concept_slide(
        prs, 1, "수직선 (Number Line)",
        "수직선은 모든 수를 한 줄에 늘어놓은 것! 온도계, 엘리베이터 모두 수직선이에요.",
        [
            "0을 중심으로 오른쪽: 양수(+), 왼쪽: 음수(-)",
            "오른쪽일수록 큰 수 (예: 3 > -2)",
            "실생활: 온도계, 층수, 시간축",
            "좌표계의 기초 — x축이 바로 수직선!",
        ],
        fig_number_line(),
    )

    # 4. 함숫값
    make_concept_slide(
        prs, 2, "함숫값 (Function)",
        "마법 상자에 숫자를 넣으면 규칙에 따라 다른 숫자가 나와요. 이게 함수!",
        [
            "함수: 입력값(x) → 규칙(f) → 출력값(y)",
            "표기법: y = f(x)  예) f(x) = x + 2",
            "하나의 입력 = 하나의 출력 (중요한 조건!)",
            "그래프로 표현 → 점들을 연결한 곡선",
        ],
    )

    # 5. 기울기
    make_concept_slide(
        prs, 3, "기울기 (Slope)",
        "언덕이 얼마나 가파른지 숫자로 표현한 것! 자전거 타기를 상상해봐요.",
        [
            "기울기 = (y₂-y₁) ÷ (x₂-x₁)",
            "양수 기울기 → 오른쪽 위 방향 ↗",
            "음수 기울기 → 오른쪽 아래 방향 ↘",
            "기울기 0 → 평평한 직선 →",
            "미분의 핵심 아이디어!",
        ],
        fig_slope(),
    )

    # 6. 이차함수
    make_concept_slide(
        prs, 4, "이차함수 (Quadratic Function)",
        "포물선 모양의 함수! 공을 던지거나 롤러코스터를 타면 이 모양이에요.",
        [
            "y = ax² + bx + c  (이차항이 있으면 이차함수)",
            "a > 0 → ∪ 모양  /  a < 0 → ∩ 모양",
            "꼭짓점 x = -b/(2a) — 가장 높거나 낮은 점",
            "대칭축을 경계로 좌우 대칭!",
            "실생활: 공의 궤적, 현수교, 반사경",
        ],
        fig_quadratic(),
    )

    # 7. 미분
    make_derivative_slide(prs)

    # 8. 적분
    make_integral_slide(prs)

    # 9. AI 튜터
    make_ai_tutor_slide(prs)

    # 10. 요약
    make_summary_slide(prs)

    prs.save(output_path)
    print(f"✅ PPT 저장 완료: {output_path}  ({len(prs.slides)}슬라이드)")
    return output_path


if __name__ == "__main__":
    build_ppt()
