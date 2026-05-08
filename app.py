"""
초등학생도 이해하는 미적분 탐험대
Streamlit + Ollama(gemma4) 기반 스토리텔링 미적분 학습 앱
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import io, textwrap, time, json

# ── Ollama ──────────────────────────────────────────────────────────────────
try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ── 한글 폰트 ──────────────────────────────────────────────────────────────
def _set_korean_font():
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
            return
    import glob
    nanum_paths = glob.glob("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
    if nanum_paths:
        font_manager.fontManager.addfont(nanum_paths[0])
        prop = font_manager.FontProperties(fname=nanum_paths[0])
        plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()

# ── 페이지 설정 ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="미적분 탐험대 🚀",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.story-box {
    background: linear-gradient(135deg,#e8f4fd,#fce4ec);
    border-left: 6px solid #2196F3;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    font-size: 1.1rem;
    line-height: 1.8;
}
.concept-card {
    background: #fff;
    border: 2px solid #4CAF50;
    border-radius: 14px;
    padding: 18px;
    margin: 10px 0;
    box-shadow: 0 4px 12px rgba(76,175,80,.15);
}
.highlight {
    background:#fff9c4;
    border-radius:6px;
    padding:2px 8px;
    font-weight:700;
    color:#e65100;
}
.chapter-title {
    font-size:2rem;
    font-weight:900;
    color:#1a237e;
    margin-bottom:4px;
}
.ai-bubble {
    background:#e3f2fd;
    border-radius:18px 18px 18px 4px;
    padding:16px 20px;
    margin:10px 0;
    border-left:4px solid #1976D2;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  챕터 정의
# ══════════════════════════════════════════════════════════════════════════════
CHAPTERS = {
    "🏠 홈": "home",
    "📏 1. 수직선": "number_line",
    "📊 2. 함숫값": "function_value",
    "📐 3. 기울기": "slope",
    "🎢 4. 이차함수": "quadratic",
    "⚡ 5. 미분은 기울기다": "derivative",
    "🏞️ 6. 적분은 넓이다": "integral",
    "🤖 AI 수학 튜터": "ai_tutor",
}


# ══════════════════════════════════════════════════════════════════════════════
#  그래프 헬퍼 함수들
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return buf


def plot_number_line(highlight=None):
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.8, 0.8)
    ax.axhline(0, color="#333", linewidth=2.5)
    for x in range(-5, 6):
        ax.plot(x, 0, "o", color="#555", markersize=7)
        ax.text(x, -0.35, str(x), ha="center", va="top", fontsize=11, color="#333")
    # 화살표
    ax.annotate("", xy=(6, 0), xytext=(5.4, 0),
                arrowprops=dict(arrowstyle="->", color="#333", lw=2))
    ax.annotate("", xy=(-6, 0), xytext=(-5.4, 0),
                arrowprops=dict(arrowstyle="->", color="#333", lw=2))
    if highlight is not None:
        ax.plot(highlight, 0, "o", color="#F44336", markersize=14, zorder=5)
        ax.text(highlight, 0.35, f"← {highlight} 위치!", ha="center",
                fontsize=12, color="#F44336", fontweight="bold")
    ax.axis("off")
    ax.set_title("수직선 (Number Line)", fontsize=14, pad=8)
    return fig


def plot_function_table_graph(func, x_vals, title="함수 그래프"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # 표
    ax_t = axes[0]
    ax_t.axis("off")
    y_vals = [func(x) for x in x_vals]
    table_data = [[str(x), str(round(y, 2))] for x, y in zip(x_vals, y_vals)]
    t = ax_t.table(cellText=table_data, colLabels=["x (입력)", "y (출력)"],
                   loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1.4, 1.8)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1976D2")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#e3f2fd")
    ax_t.set_title("입출력 표", fontsize=13, pad=6)
    # 그래프
    ax_g = axes[1]
    xs = np.linspace(min(x_vals) - 0.5, max(x_vals) + 0.5, 300)
    ys = [func(x) for x in xs]
    ax_g.plot(xs, ys, color="#1976D2", linewidth=2.5)
    ax_g.scatter(x_vals, y_vals, color="#F44336", s=70, zorder=5)
    ax_g.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax_g.axvline(0, color="#888", linewidth=0.8, linestyle="--")
    ax_g.set_title(title, fontsize=13)
    ax_g.grid(True, alpha=0.3)
    ax_g.set_xlabel("x")
    ax_g.set_ylabel("y = f(x)")
    plt.tight_layout()
    return fig


def plot_slope(x1, y1, x2, y2):
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.linspace(x1 - 2, x2 + 2, 200)
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    ys = slope * xs + intercept
    ax.plot(xs, ys, color="#1976D2", linewidth=2.5, label=f"기울기 = {slope:.2f}")
    ax.plot([x1, x2], [y1, y2], "ro", markersize=10)
    ax.annotate("", xy=(x2, y1), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=2))
    ax.annotate("", xy=(x2, y2), xytext=(x2, y1),
                arrowprops=dict(arrowstyle="->", color="#F44336", lw=2))
    ax.text((x1 + x2) / 2, y1 - 0.6, f"가로 이동: {x2-x1}",
            ha="center", fontsize=11, color="#4CAF50")
    ax.text(x2 + 0.2, (y1 + y2) / 2, f"세로 이동: {y2-y1}",
            ha="left", fontsize=11, color="#F44336")
    slope_str = f"{slope:.2f}" if slope == slope else "?"
    ax.set_title(f"기울기 = 세로 변화 ÷ 가로 변화 = {y2-y1}/{x2-x1} = {slope_str}",
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8)
    return fig


def plot_quadratic(a=1, b=0, c=0):
    x = np.linspace(-5, 5, 400)
    y = a * x**2 + b * x + c
    vertex_x = -b / (2 * a)
    vertex_y = a * vertex_x**2 + b * vertex_x + c
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color="#9C27B0", linewidth=2.8,
            label=f"y = {a}x² + {b}x + {c}")
    ax.plot(vertex_x, vertex_y, "y*", markersize=18, zorder=5,
            label=f"꼭짓점 ({vertex_x:.1f}, {vertex_y:.1f})")
    ax.axhline(0, color="#888", lw=0.8, linestyle="--")
    ax.axvline(0, color="#888", lw=0.8, linestyle="--")
    ax.set_ylim(min(y) - 1, max(y) + 1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title("이차함수 그래프", fontsize=13)
    ax.set_xlabel("x"), ax.set_ylabel("y")
    return fig


def plot_derivative_limit(func, dfunc, x0=1.5, steps=5):
    """극한으로 미분 이해하기 - 할선이 접선으로 수렴"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.linspace(-0.3, 3.5, 400)
    y = [func(xi) for xi in x]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, steps))
    h_vals = [2.0, 1.0, 0.5, 0.2, 0.05]

    for ax_idx, ax in enumerate(axes):
        ax.plot(x, y, color="#1976D2", linewidth=2.5, label="f(x) = x²", zorder=3)
        ax.plot(x0, func(x0), "ko", markersize=8, zorder=5)

        if ax_idx == 0:
            for i, h in enumerate(h_vals):
                m = (func(x0 + h) - func(x0)) / h
                b = func(x0) - m * x0
                xs_line = np.linspace(x0 - 0.5, x0 + h + 0.3, 50)
                ys_line = m * xs_line + b
                ax.plot(xs_line, ys_line, color=colors[i], linewidth=1.8,
                        label=f"h={h:.2f}, 기울기={m:.2f}", alpha=0.85)
                ax.plot(x0 + h, func(x0 + h), "o", color=colors[i], markersize=7)
            ax.set_title("h를 작게 할수록 접선에 가까워짐", fontsize=12)
            ax.legend(fontsize=8.5, loc="upper left")
        else:
            # 접선
            m_tangent = dfunc(x0)
            b_tangent = func(x0) - m_tangent * x0
            xs_t = np.linspace(x0 - 1.5, x0 + 1.5, 100)
            ys_t = m_tangent * xs_t + b_tangent
            ax.plot(xs_t, ys_t, color="#F44336", linewidth=2.8,
                    label=f"접선 기울기 = f'({x0}) = {m_tangent:.2f}", zorder=4)
            ax.set_title(f"최종 접선: 미분값 f'({x0}) = {m_tangent:.2f}", fontsize=12)
            ax.legend(fontsize=10)

        ax.set_xlim(-0.3, 3.5)
        ax.set_ylim(-0.5, 10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="#888", lw=0.8)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_xlabel("x"), ax.set_ylabel("y")

    plt.tight_layout()
    return fig


def plot_riemann(func, a, b, n, method="mid"):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.linspace(a - 0.5, b + 0.5, 400)
    y = [func(xi) for xi in x]
    ax.plot(x, y, color="#1976D2", linewidth=2.5, zorder=4, label="f(x)")

    dx = (b - a) / n
    total_area = 0.0
    xs_rect = np.linspace(a, b - dx, n)
    for xi in xs_rect:
        if method == "left":
            height = func(xi)
        elif method == "right":
            height = func(xi + dx)
        else:
            height = func(xi + dx / 2)
        total_area += height * dx
        rect = patches.Rectangle((xi, 0), dx, height,
                                  linewidth=0.8, edgecolor="#333",
                                  facecolor="#4CAF50", alpha=0.45)
        ax.add_patch(rect)

    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlim(a - 0.5, b + 0.5)
    ax.set_ylim(0, max(y) * 1.15)
    ax.set_title(f"리만 합으로 넓이 구하기 (n={n}개 직사각형, 합계 ≈ {total_area:.4f})",
                 fontsize=12)
    ax.set_xlabel("x"), ax.set_ylabel("y = f(x)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    return fig


def plot_definite_integral(func, a, b, title="정적분 = 곡선 아래 넓이"):
    x_fill = np.linspace(a, b, 300)
    y_fill = [func(xi) for xi in x_fill]
    x_all = np.linspace(a - 0.5, b + 0.5, 400)
    y_all = [func(xi) for xi in x_all]

    from scipy import integrate as sci_int
    area, _ = sci_int.quad(func, a, b)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_all, y_all, color="#1976D2", linewidth=2.5, label="f(x)")
    ax.fill_between(x_fill, 0, y_fill, color="#FF9800", alpha=0.5,
                    label=f"넓이 = {area:.4f}")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("x"), ax.set_ylabel("y")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  AI 튜터 함수
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """당신은 '수학 탐험대 AI 튜터'입니다.
초등학생도 이해할 수 있도록 미적분 개념을 쉽고 재미있게 설명하세요.
규칙:
1. 어려운 수학 기호 대신 일상 언어와 비유를 사용하세요.
2. 예시는 항상 아이들이 아는 것(놀이터, 자전거, 피자, 물 등)으로 드세요.
3. 한 번에 한 가지 개념만 설명하세요.
4. 응원과 격려의 말을 꼭 포함하세요.
5. 한국어로 답변하세요.
6. 수식이 필요하면 최대한 단순하게 쓰세요."""

def ask_ollama(question: str, model: str, history: list) -> str:
    if not OLLAMA_AVAILABLE:
        return "⚠️ ollama 패키지가 설치되지 않았습니다. `pip install ollama`를 실행하세요."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": question})
    try:
        response = _ollama.chat(model=model, messages=messages)
        return response["message"]["content"]
    except Exception as e:
        err = str(e)
        if "model" in err.lower() and ("not found" in err.lower() or "pull" in err.lower()):
            return f"⚠️ 모델 '{model}'을 찾을 수 없습니다. 터미널에서 `ollama pull {model}`을 실행해 주세요."
        return f"⚠️ Ollama 오류: {err}"


# ══════════════════════════════════════════════════════════════════════════════
#  챕터별 페이지 렌더링
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.markdown("""
<div style="text-align:center; padding:30px 0;">
  <h1 style="font-size:3rem; color:#1a237e;">🚀 미적분 탐험대</h1>
  <p style="font-size:1.4rem; color:#555;">초등학생도 이해하는 미분·적분 여행</p>
</div>
""", unsafe_allow_html=True)

    cols = st.columns(3)
    cards = [
        ("🎯", "단계별 학습", "수직선 → 함수 → 기울기 → 미분 → 적분,\n순서대로 차근차근!"),
        ("🎨", "시각화", "모든 개념을 그래프와 애니메이션으로\n눈으로 확인해요."),
        ("🤖", "AI 튜터", "모르는 게 생기면 AI에게 물어보세요!\ngemma4 모델이 친절하게 답변해요."),
    ]
    for col, (icon, title, desc) in zip(cols, cards):
        col.markdown(f"""
<div class="concept-card" style="text-align:center;">
  <div style="font-size:2.5rem;">{icon}</div>
  <h3 style="color:#1976D2;">{title}</h3>
  <p style="color:#555;">{desc}</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 학습 여정")
    steps = [
        ("1", "📏 수직선", "숫자들의 집"),
        ("2", "📊 함숫값", "마법 상자 함수"),
        ("3", "📐 기울기", "얼마나 가파를까?"),
        ("4", "🎢 이차함수", "포물선의 세계"),
        ("5", "⚡ 미분", "순간의 기울기"),
        ("6", "🏞️ 적분", "넓이를 구하자"),
    ]
    cols2 = st.columns(6)
    for col, (num, icon_title, sub) in zip(cols2, steps):
        col.markdown(f"""
<div style="text-align:center; background:#f5f5f5; border-radius:10px; padding:12px;">
  <div style="font-size:1.8rem;">{icon_title.split()[0]}</div>
  <div style="font-weight:700; color:#1a237e; font-size:.95rem;">{icon_title[2:]}</div>
  <div style="font-size:.8rem; color:#777;">{sub}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 왼쪽 사이드바에서 원하는 챕터를 선택하세요!")


def page_number_line():
    st.markdown('<div class="chapter-title">📏 1장. 수직선</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🌟 <b>오늘의 이야기</b><br>
수직선이 없는 세상을 상상해봐요! 우리는 어떻게 "나는 3층에 살아요" 또는
"영하 5도야 너무 추워!"라고 말할 수 있을까요?<br>
수직선은 <span class="highlight">모든 수학의 시작점</span>이에요.
숫자들이 일렬로 서 있는 긴 복도라고 생각해봐요! 🏠
</div>
""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 개념 이해", "🎮 직접 해보기", "🧩 퀴즈"])

    with tab1:
        st.markdown("""
<div class="concept-card">
<h4>수직선이란?</h4>
<ul>
<li>숫자를 <b>왼쪽 → 오른쪽</b> 순서로 늘어놓은 선이에요</li>
<li>중간에 <b>0 (영)</b>이 있어요 — 기준점!</li>
<li>0보다 오른쪽: <b>양수</b> (+1, +2, +3 ...)</li>
<li>0보다 왼쪽: <b>음수</b> (-1, -2, -3 ...)</li>
<li>숫자가 클수록 오른쪽에 있어요</li>
</ul>
</div>
""", unsafe_allow_html=True)
        fig = plot_number_line()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
<div class="story-box">
🎯 <b>일상 속 수직선</b><br>
• 온도계 → 위아래로 된 수직선!<br>
• 엘리베이터 버튼 → B1(지하1층)은 음수, 위층은 양수<br>
• 스포츠 점수판 → 더 큰 숫자가 오른쪽에!
</div>
""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 내가 원하는 숫자를 수직선에 표시해봐요!")
        val = st.slider("숫자를 선택하세요", -5, 5, 2)
        fig2 = plot_number_line(highlight=val)
        st.pyplot(fig2)
        plt.close(fig2)
        if val > 0:
            st.success(f"✅ {val}은 **양수**예요! 0보다 오른쪽에 있어요.")
        elif val < 0:
            st.warning(f"❄️ {val}은 **음수**예요! 0보다 왼쪽에 있어요.")
        else:
            st.info("🎯 0은 수직선의 중심, 기준점이에요!")

        st.markdown("#### 두 수 중 어느 것이 더 클까요?")
        c1, c2 = st.columns(2)
        num_a = c1.number_input("첫 번째 수", value=3, step=1, key="na")
        num_b = c2.number_input("두 번째 수", value=-1, step=1, key="nb")
        if st.button("비교하기!"):
            if num_a > num_b:
                st.success(f"🏆 {num_a} > {num_b} : {num_a}이 더 커요! (수직선에서 더 오른쪽)")
            elif num_a < num_b:
                st.success(f"🏆 {num_a} < {num_b} : {num_b}이 더 커요! (수직선에서 더 오른쪽)")
            else:
                st.info("⚖️ 두 수가 같아요!")

    with tab3:
        st.markdown("#### 🧩 수직선 퀴즈!")
        q1 = st.radio("-3과 2 중 어느 수가 수직선에서 오른쪽에 있나요?",
                      ["−3", "2", "둘 다 같다"], key="q_nl1")
        if st.button("정답 확인", key="btn_nl1"):
            if q1 == "2":
                st.success("🎉 정답! 2는 -3보다 오른쪽에 있어요. 2 > -3 이니까요!")
            else:
                st.error("다시 생각해봐요. 수직선에서 오른쪽일수록 큰 수예요!")

        st.markdown("---")
        q2 = st.radio("0은 양수인가요, 음수인가요?",
                      ["양수", "음수", "둘 다 아니에요"], key="q_nl2")
        if st.button("정답 확인", key="btn_nl2"):
            if q2 == "둘 다 아니에요":
                st.success("🎉 맞아요! 0은 양수도 음수도 아닌 특별한 수예요!")
            else:
                st.error("0은 양수도 음수도 아니에요. 수직선의 가운데 기준점이에요!")


def page_function_value():
    st.markdown('<div class="chapter-title">📊 2장. 함숫값</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🎰 <b>오늘의 이야기 — 마법 상자</b><br>
여기 신비한 마법 상자가 있어요! 이 상자에 숫자를 넣으면,
상자 안에서 어떤 <b>규칙</b>에 따라 다른 숫자로 바뀌어 나와요.<br>
예를 들어 "항상 2를 더하는 상자"가 있다면:<br>
➡️ 3을 넣으면 → <span class="highlight">5</span>가 나와요!<br>
➡️ 10을 넣으면 → <span class="highlight">12</span>가 나와요!<br>
이 상자가 바로 <b>함수</b>예요! 📦✨
</div>
""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 개념 이해", "🎮 함수 탐험기", "📈 그래프 그리기"])

    with tab1:
        st.markdown("""
<div class="concept-card">
<h4>함수(Function)란?</h4>
<ul>
<li>입력값(x)을 넣으면 출력값(y)이 나오는 <b>규칙</b></li>
<li>표기법: y = f(x) → "x를 넣었을 때 f의 결과"</li>
<li>한 입력값에는 반드시 <b>하나의</b> 출력값만 있어요</li>
</ul>
</div>
""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.markdown("""
**예시 1: y = x + 2**
| x (입력) | y (출력) |
|:---:|:---:|
| 1 | 3 |
| 2 | 4 |
| 5 | 7 |
| -1 | 1 |
""")
        col2.markdown("""
**예시 2: y = x × 2**
| x (입력) | y (출력) |
|:---:|:---:|
| 1 | 2 |
| 3 | 6 |
| 5 | 10 |
| 0 | 0 |
""")

    with tab2:
        st.markdown("#### 나만의 함수 만들기!")
        func_type = st.selectbox("함수 규칙 선택", [
            "y = x + a (더하기)",
            "y = x × a (곱하기)",
            "y = x² (제곱)",
            "y = 2x + 3 (일차함수)",
        ])
        x_input = st.number_input("x 값을 입력하세요", value=3.0, step=0.5)

        if func_type == "y = x + a (더하기)":
            a = st.slider("a 값", -10, 10, 2)
            result = x_input + a
            formula = f"y = {x_input} + {a}"
        elif func_type == "y = x × a (곱하기)":
            a = st.slider("a 값", -5, 10, 3)
            result = x_input * a
            formula = f"y = {x_input} × {a}"
        elif func_type == "y = x² (제곱)":
            result = x_input ** 2
            formula = f"y = {x_input}²"
            a = None
        else:
            result = 2 * x_input + 3
            formula = f"y = 2 × {x_input} + 3"
            a = None

        st.markdown(f"""
<div class="concept-card" style="text-align:center; font-size:1.4rem;">
📥 {formula} = <span class="highlight">{result:.2f}</span>
</div>
""", unsafe_allow_html=True)

    with tab3:
        st.markdown("#### 함수 그래프 그리기!")
        f_choice = st.selectbox("그릴 함수", [
            "y = x", "y = 2x", "y = x + 3", "y = x²", "y = 2x + 1"
        ], key="f_graph")
        func_map = {
            "y = x": (lambda x: x, "y = x"),
            "y = 2x": (lambda x: 2 * x, "y = 2x"),
            "y = x + 3": (lambda x: x + 3, "y = x + 3"),
            "y = x²": (lambda x: x ** 2, "y = x²"),
            "y = 2x + 1": (lambda x: 2 * x + 1, "y = 2x + 1"),
        }
        fn, fn_title = func_map[f_choice]
        x_vals = list(range(-4, 5))
        fig = plot_function_table_graph(fn, x_vals, fn_title)
        st.pyplot(fig)
        plt.close(fig)


def page_slope():
    st.markdown('<div class="chapter-title">📐 3장. 기울기</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🏔️ <b>오늘의 이야기 — 언덕 오르기</b><br>
자전거를 타고 언덕을 오른다고 생각해봐요!<br>
어떤 언덕은 <b>완만</b>하고 (기울기가 작아요),<br>
어떤 언덕은 <b>가파르죠</b> (기울기가 커요).<br>
기울기는 "얼마나 가파른가?"를 숫자로 표현한 거예요! 🚵
</div>
""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 개념 이해", "🎮 기울기 탐험", "🧩 실생활 예시"])

    with tab1:
        st.markdown("""
<div class="concept-card">
<h4>기울기 공식</h4>
<p style="font-size:1.5rem; text-align:center; color:#9C27B0;">
기울기 = <b>세로 변화</b> ÷ <b>가로 변화</b>
</p>
<p style="font-size:1.2rem; text-align:center; color:#555;">
= (y₂ - y₁) ÷ (x₂ - x₁)
</p>
<ul>
<li>기울기 > 0 : 오른쪽으로 올라가는 선 ↗</li>
<li>기울기 < 0 : 오른쪽으로 내려가는 선 ↘</li>
<li>기울기 = 0 : 완전히 평평한 선 →</li>
<li>기울기가 클수록 더 가파른 선!</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 두 점으로 기울기 구하기!")
        c1, c2 = st.columns(2)
        x1 = c1.number_input("점 A의 x 좌표", value=1.0, step=0.5, key="sx1")
        y1 = c1.number_input("점 A의 y 좌표", value=1.0, step=0.5, key="sy1")
        x2 = c2.number_input("점 B의 x 좌표", value=4.0, step=0.5, key="sx2")
        y2 = c2.number_input("점 B의 y 좌표", value=7.0, step=0.5, key="sy2")

        if abs(x2 - x1) < 0.001:
            st.error("⚠️ 두 점의 x 좌표가 같으면 기울기를 구할 수 없어요!")
        else:
            slope = (y2 - y1) / (x2 - x1)
            st.markdown(f"""
<div class="concept-card" style="text-align:center; font-size:1.3rem;">
기울기 = ({y2:.1f} - {y1:.1f}) ÷ ({x2:.1f} - {x1:.1f}) =
<span class="highlight">{slope:.2f}</span>
</div>
""", unsafe_allow_html=True)
            fig = plot_slope(x1, y1, x2, y2)
            st.pyplot(fig)
            plt.close(fig)

            if slope > 2:
                st.info("🏔️ 기울기가 꽤 가파르네요!")
            elif slope > 0:
                st.info("🐢 완만하게 올라가는 언덕이에요.")
            elif slope < 0:
                st.info("🛷 내려가는 방향이에요!")
            else:
                st.info("➡️ 완전히 평평해요.")

    with tab3:
        st.markdown("""
#### 실생활 속 기울기
""")
        examples = [
            ("🛝 미끄럼틀", "기울기가 클수록 더 빠르게 내려와요!"),
            ("🏗️ 지붕", "비가 흘러내리려면 적당한 기울기가 필요해요"),
            ("🛣️ 도로 경사", "오르막길 표지판에 있는 % = 기울기!"),
            ("📈 주가 차트", "올라가면 양의 기울기, 내려가면 음의 기울기"),
        ]
        cols = st.columns(2)
        for i, (icon, desc) in enumerate(examples):
            cols[i % 2].markdown(f"""
<div class="concept-card">
<b>{icon}</b><br>{desc}
</div>""", unsafe_allow_html=True)


def page_quadratic():
    st.markdown('<div class="chapter-title">🎢 4장. 이차함수</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🎢 <b>오늘의 이야기 — 롤러코스터</b><br>
롤러코스터는 왜 그렇게 생겼을까요? 공을 던지면 왜 포물선을 그릴까요?<br>
이 모든 것이 <b>이차함수</b>의 모습이에요!<br>
y = ax² + bx + c 라는 공식으로 이 아름다운 곡선을 만들 수 있어요. 🎯
</div>
""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 개념 이해", "🎮 이차함수 조각가", "🌍 실생활 예시"])

    with tab1:
        col1, col2 = st.columns(2)
        col1.markdown("""
<div class="concept-card">
<h4>이차함수란?</h4>
<p>y = ax² + bx + c</p>
<ul>
<li><b>a</b> : 포물선 방향 결정<br>a > 0 → ∪ (위로 열림)<br>a < 0 → ∩ (아래로 열림)</li>
<li><b>꼭짓점</b> : 가장 높거나 낮은 점</li>
<li><b>대칭축</b> : 꼭짓점을 지나는 세로선</li>
</ul>
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="concept-card">
<h4>예시 함수들</h4>
<ul>
<li>y = x² → 기본 포물선 (∪)</li>
<li>y = -x² → 뒤집힌 포물선 (∩)</li>
<li>y = 2x² → 더 좁은 포물선</li>
<li>y = x² - 4 → 아래로 이동</li>
<li>y = (x-3)² → 오른쪽으로 이동</li>
</ul>
</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### a, b, c를 조절해서 이차함수를 만들어보세요!")
        c1, c2, c3 = st.columns(3)
        a = c1.slider("a (포물선 방향·넓이)", -3.0, 3.0, 1.0, 0.5)
        b = c2.slider("b (좌우 이동)", -5.0, 5.0, 0.0, 0.5)
        c = c3.slider("c (위아래 이동)", -5.0, 5.0, 0.0, 0.5)

        if abs(a) < 0.01:
            st.warning("a가 0에 너무 가까우면 이차함수가 아니에요!")
        else:
            vertex_x = -b / (2 * a)
            vertex_y = a * vertex_x ** 2 + b * vertex_x + c
            st.markdown(f"""
<div class="concept-card" style="text-align:center;">
y = {a}x² + {b}x + {c} &nbsp;&nbsp;|&nbsp;&nbsp;
꼭짓점: ({vertex_x:.2f}, {vertex_y:.2f}) &nbsp;&nbsp;|&nbsp;&nbsp;
{'∪ 위로 열림' if a > 0 else '∩ 아래로 열림'}
</div>""", unsafe_allow_html=True)
            fig = plot_quadratic(a, b, c)
            st.pyplot(fig)
            plt.close(fig)

    with tab3:
        st.markdown("""
#### 이차함수가 실생활 어디에 있을까요?
""")
        examples2 = [
            ("⚽ 공의 궤적", "축구공을 차면 포물선을 그려요!", "y = -x² + 5x"),
            ("🌉 현수교", "다리를 잇는 줄의 모양이에요", "y = 0.1x²"),
            ("🔦 손전등", "반사판이 포물선 모양이에요", "y = x²"),
            ("🚀 로켓", "연료 없으면 포물선으로 떨어져요", "y = -0.5x² + 10x"),
        ]
        for icon, desc, formula in examples2:
            st.markdown(f"""
<div class="concept-card">
<b>{icon} {desc}</b><br>
<code>{formula}</code>
</div>""", unsafe_allow_html=True)


def page_derivative():
    st.markdown('<div class="chapter-title">⚡ 5장. 미분은 기울기다</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🏎️ <b>오늘의 이야기 — 순간 속도</b><br>
자동차가 달리는데, <b>딱 이 순간</b>의 속도는 얼마일까요?<br>
속도계 바늘이 60km/h를 가리키고 있다면,
그건 "지금 이 순간의 속도"예요.<br>
<b>미분</b>이 바로 이것이에요 — 곡선의 <span class="highlight">순간 기울기</span>를 구하는 것! 🚀
</div>
""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 개념", "🔬 극한 탐구", "⚡ 미분 연습", "📐 미분 공식"])

    with tab1:
        st.markdown("""
<div class="concept-card">
<h4>미분의 핵심 아이디어</h4>
<ol>
<li>곡선 위 두 점 사이의 기울기(할선)를 구해요</li>
<li>두 점을 점점 가까이 이동시켜요</li>
<li>두 점이 거의 같아지면, 그게 바로 <b>접선의 기울기</b></li>
<li>이 접선의 기울기 = <b>미분값</b> f'(x)</li>
</ol>
<p style="text-align:center; font-size:1.3rem; color:#9C27B0;">
f'(x) = lim<sub>h→0</sub> [f(x+h) - f(x)] / h
</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="story-box">
💡 <b>쉽게 이해하기</b><br>
"순간 속도"를 재려면?<br>
→ 아주 짧은 시간(h) 동안 이동한 거리를 측정해요<br>
→ h를 0에 가깝게 줄이면 → 진짜 순간 속도!<br>
이게 바로 극한(limit)이에요. 수학의 마법! ✨
</div>
""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### h를 줄여가며 극한 체험하기")
        st.markdown("f(x) = x² 일 때, x=2에서의 미분값을 구해봐요")

        x0_val = st.slider("x₀ 위치", 0.5, 3.0, 2.0, 0.5)
        h_val = st.select_slider("h 값 (점점 작게!)",
                                  options=[2.0, 1.0, 0.5, 0.2, 0.1, 0.01, 0.001],
                                  value=1.0)
        f = lambda x: x ** 2
        slope_approx = (f(x0_val + h_val) - f(x0_val)) / h_val
        true_slope = 2 * x0_val

        st.markdown(f"""
<div class="concept-card" style="text-align:center; font-size:1.2rem;">
h = {h_val} 일 때 기울기 ≈ <b>{slope_approx:.6f}</b><br>
진짜 미분값 f'({x0_val}) = 2×{x0_val} = <b style="color:#F44336;">{true_slope:.4f}</b><br>
오차: {abs(slope_approx - true_slope):.8f}
</div>
""", unsafe_allow_html=True)

        if abs(slope_approx - true_slope) < 0.01:
            st.success(f"🎉 h={h_val}로 거의 정확해졌어요! 미분값에 수렴하고 있어요!")
        else:
            st.info(f"h를 더 작게 줄여봐요! 미분값 {true_slope}에 점점 가까워져요.")

        fig = plot_derivative_limit(lambda x: x**2, lambda x: 2*x, x0=x0_val)
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.markdown("#### 다양한 함수의 미분 계산기")
        fn2_choice = st.selectbox("함수 선택", [
            "f(x) = x²", "f(x) = x³", "f(x) = 3x² + 2x",
            "f(x) = x² - 4x + 3"
        ])
        x_calc = st.slider("x 값", -3.0, 3.0, 1.5, 0.5)

        fn2_map = {
            "f(x) = x²": (lambda x: x**2, lambda x: 2*x, "f'(x) = 2x"),
            "f(x) = x³": (lambda x: x**3, lambda x: 3*x**2, "f'(x) = 3x²"),
            "f(x) = 3x² + 2x": (lambda x: 3*x**2+2*x, lambda x: 6*x+2, "f'(x) = 6x + 2"),
            "f(x) = x² - 4x + 3": (lambda x: x**2-4*x+3, lambda x: 2*x-4, "f'(x) = 2x - 4"),
        }
        fn_v, df_v, df_formula = fn2_map[fn2_choice]

        col1, col2 = st.columns(2)
        col1.metric("함수값 f(x)", f"{fn_v(x_calc):.4f}")
        col2.metric("미분값 f'(x) — 순간 기울기", f"{df_v(x_calc):.4f}")
        st.markdown(f"**공식:** {df_formula}")

        x_arr = np.linspace(-3.5, 3.5, 300)
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.plot(x_arr, [fn_v(xi) for xi in x_arr], color="#1976D2", lw=2.5, label=fn2_choice)
        # 접선
        m_t = df_v(x_calc)
        b_t = fn_v(x_calc) - m_t * x_calc
        x_tan = np.linspace(x_calc - 1.2, x_calc + 1.2, 50)
        ax1.plot(x_tan, m_t * x_tan + b_t, color="#F44336", lw=2, linestyle="--",
                 label=f"접선 기울기={m_t:.2f}")
        ax1.plot(x_calc, fn_v(x_calc), "ko", markersize=8)
        ax1.set_title("원래 함수 + 접선", fontsize=12)
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color="#888", lw=0.8); ax1.axvline(0, color="#888", lw=0.8)

        ax2.plot(x_arr, [df_v(xi) for xi in x_arr], color="#9C27B0", lw=2.5,
                 label=df_formula)
        ax2.plot(x_calc, df_v(x_calc), "ro", markersize=8)
        ax2.set_title("도함수 (미분한 함수)", fontsize=12)
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color="#888", lw=0.8); ax2.axvline(0, color="#888", lw=0.8)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with tab4:
        st.markdown("""
<div class="concept-card">
<h4>기본 미분 공식</h4>
<table style="width:100%; font-size:1.1rem; border-collapse:collapse;">
<tr style="background:#1976D2; color:white;">
  <th style="padding:8px;">함수 f(x)</th><th style="padding:8px;">미분 f'(x)</th><th style="padding:8px;">예시</th>
</tr>
<tr style="background:#e3f2fd;"><td style="padding:6px; text-align:center;">xⁿ</td><td style="padding:6px; text-align:center;">n·xⁿ⁻¹</td><td style="padding:6px;">x³ → 3x²</td></tr>
<tr><td style="padding:6px; text-align:center;">상수 c</td><td style="padding:6px; text-align:center;">0</td><td style="padding:6px;">5 → 0</td></tr>
<tr style="background:#e3f2fd;"><td style="padding:6px; text-align:center;">x</td><td style="padding:6px; text-align:center;">1</td><td style="padding:6px;">x → 1</td></tr>
<tr><td style="padding:6px; text-align:center;">x²</td><td style="padding:6px; text-align:center;">2x</td><td style="padding:6px;">x²를 미분하면 2x</td></tr>
<tr style="background:#e3f2fd;"><td style="padding:6px; text-align:center;">3x² + 2x</td><td style="padding:6px; text-align:center;">6x + 2</td><td style="padding:6px;">각 항을 따로 미분!</td></tr>
</table>
</div>
""", unsafe_allow_html=True)


def page_integral():
    st.markdown('<div class="chapter-title">🏞️ 6장. 적분은 넓이다</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🍕 <b>오늘의 이야기 — 피자 넓이 구하기</b><br>
이상하게 생긴 땅의 넓이를 구하고 싶어요!<br>
직사각형은 쉽지만... 곡선으로 이루어진 도형은요? 🤔<br>
비법: 아주 얇은 <b>직사각형</b> 여러 개로 나눠서 더하면 돼요!<br>
직사각형이 많을수록 → 더 정확해요!<br>
무한히 많이 나누면 → 완벽한 넓이! 이게 바로 <span class="highlight">적분</span>이에요! 🎯
</div>
""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 개념", "📦 리만 합", "📐 정적분 계산", "🌍 응용"])

    with tab1:
        col1, col2 = st.columns(2)
        col1.markdown("""
<div class="concept-card">
<h4>적분의 아이디어</h4>
<ol>
<li>곡선 아래를 <b>아주 얇은 직사각형</b>으로 채워요</li>
<li>직사각형 넓이 = 높이(f(x)) × 폭(Δx)</li>
<li>모든 직사각형을 더해요 → 리만 합</li>
<li>폭을 0에 가깝게 → 진짜 넓이!</li>
</ol>
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="concept-card">
<h4>정적분 표기</h4>
<p style="font-size:1.6rem; text-align:center; color:#E65100;">
∫<sub>a</sub><sup>b</sup> f(x) dx
</p>
<ul>
<li>a : 시작점 (왼쪽)</li>
<li>b : 끝점 (오른쪽)</li>
<li>f(x) : 함수 (곡선의 높이)</li>
<li>dx : 아주 작은 폭</li>
</ul>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="story-box">
💡 <b>미분과 적분의 관계 — 미적분학의 기본 정리</b><br>
미분과 적분은 서로 <b>반대</b> 방향이에요!<br>
→ 미분: 함수 → 기울기 함수<br>
→ 적분: 기울기 함수 → 원래 함수 (역방향!)<br>
마치 덧셈과 뺄셈처럼, 곱셈과 나눗셈처럼요! ⚖️
</div>
""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 리만 합으로 넓이 구하기")
        fn3_choice = st.selectbox("함수 선택", [
            "f(x) = x²", "f(x) = x³", "f(x) = 2x + 1", "f(x) = x² + 1"
        ], key="riemann_fn")
        fn3_map = {
            "f(x) = x²": lambda x: x**2,
            "f(x) = x³": lambda x: x**3,
            "f(x) = 2x + 1": lambda x: 2*x + 1,
            "f(x) = x² + 1": lambda x: x**2 + 1,
        }
        fn3 = fn3_map[fn3_choice]

        c1r, c2r, c3r = st.columns(3)
        a_r = c1r.number_input("시작점 a", value=0.0, step=0.5, key="ra")
        b_r = c2r.number_input("끝점 b", value=3.0, step=0.5, key="rb")
        n_r = c3r.slider("직사각형 개수 n", 1, 100, 10)

        if a_r >= b_r:
            st.error("끝점 b는 시작점 a보다 커야 해요!")
        else:
            method = st.radio("넓이 측정 방법", ["mid (중간)", "left (왼쪽)", "right (오른쪽)"],
                              horizontal=True)
            m = method.split()[0]
            fig4 = plot_riemann(fn3, a_r, b_r, n_r, method=m)
            st.pyplot(fig4)
            plt.close(fig4)

            st.info(f"💡 n을 크게 할수록 더 정확한 넓이를 구할 수 있어요! "
                    f"n={n_r}개 직사각형으로 근사했어요.")

    with tab3:
        st.markdown("#### 정적분 계산기 (정확한 넓이)")
        fn4_choice = st.selectbox("함수 선택", [
            "f(x) = x²", "f(x) = 2x", "f(x) = x² + 1",
            "f(x) = x³", "f(x) = 3x² - 2"
        ], key="int_fn")
        fn4_map = {
            "f(x) = x²": (lambda x: x**2, "x³/3", lambda a, b: (b**3-a**3)/3),
            "f(x) = 2x": (lambda x: 2*x, "x²", lambda a, b: b**2-a**2),
            "f(x) = x² + 1": (lambda x: x**2+1, "x³/3 + x", lambda a, b: (b**3-a**3)/3+(b-a)),
            "f(x) = x³": (lambda x: x**3, "x⁴/4", lambda a, b: (b**4-a**4)/4),
            "f(x) = 3x² - 2": (lambda x: 3*x**2-2, "x³ - 2x", lambda a, b: (b**3-a**3)-2*(b-a)),
        }
        fn4, anti_formula, exact_fn = fn4_map[fn4_choice]

        c1i, c2i = st.columns(2)
        a_i = c1i.number_input("a (아래 한계)", value=0.0, step=0.5, key="ia")
        b_i = c2i.number_input("b (위 한계)", value=2.0, step=0.5, key="ib")

        if a_i >= b_i:
            st.error("b > a 이어야 해요!")
        else:
            area = exact_fn(a_i, b_i)
            st.markdown(f"""
<div class="concept-card" style="text-align:center; font-size:1.3rem;">
∫<sub>{a_i}</sub><sup>{b_i}</sup> {fn4_choice[5:]} dx<br>
= [{anti_formula}]<sub>{a_i}</sub><sup>{b_i}</sup><br>
= <span class="highlight" style="font-size:1.6rem;">{area:.4f}</span>
</div>
""", unsafe_allow_html=True)
            fig5 = plot_definite_integral(fn4, a_i, b_i,
                                          f"∫ {fn4_choice[5:]} dx 의 넓이 = {area:.4f}")
            st.pyplot(fig5)
            plt.close(fig5)

    with tab4:
        st.markdown("#### 정적분의 실생활 응용")
        apps = [
            ("💧 물의 양", "수도꼭지에서 1시간 동안 나온 물의 총량\n= 유량(속도) 함수를 적분!", "#e3f2fd"),
            ("🚗 이동 거리", "자동차 속도 함수를 적분하면\n= 총 이동 거리!", "#fce4ec"),
            ("🌡️ 평균 온도", "하루 온도 변화를 적분하면\n= 하루 평균 온도 계산 가능!", "#f3e5f5"),
            ("🏗️ 구조물 무게", "건물 재료 밀도를 적분하면\n= 전체 무게 계산!", "#e8f5e9"),
        ]
        cols = st.columns(2)
        for i, (icon, desc, bg) in enumerate(apps):
            cols[i % 2].markdown(f"""
<div class="concept-card" style="background:{bg};">
<b>{icon}</b><br>
<pre style="background:transparent;border:none;font-size:.9rem;">{desc}</pre>
</div>""", unsafe_allow_html=True)


def page_ai_tutor():
    st.markdown('<div class="chapter-title">🤖 AI 수학 튜터</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🤖 <b>gemma4 AI 튜터에게 무엇이든 물어보세요!</b><br>
미적분에 대해 궁금한 것, 이해가 안 되는 것,
더 알고 싶은 것 무엇이든 질문해봐요!<br>
AI 튜터는 초등학생도 이해할 수 있게 친절하게 설명해줘요. 😊
</div>
""", unsafe_allow_html=True)

    if not OLLAMA_AVAILABLE:
        st.error("""
⚠️ **ollama 패키지가 없어요!**

터미널에서 다음을 실행하세요:
```bash
pip install ollama
```

그리고 Ollama 서버가 실행 중인지 확인하세요:
```bash
ollama serve
```
""")
        return

    # 모델 선택
    col_m, col_c = st.columns([3, 1])
    model_choice = col_m.selectbox(
        "사용할 모델",
        ["gemma4:e4b", "gemma4:26b"],
        help="e4b는 빠르고, 26b는 더 자세한 설명을 해줘요"
    )
    if col_c.button("대화 초기화 🗑️"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

    # 빠른 질문 버튼
    st.markdown("#### 💬 빠른 질문")
    quick_questions = [
        "미분이 뭔지 쉽게 설명해줘!",
        "적분을 피자로 설명해줄 수 있어?",
        "기울기가 뭐야?",
        "미분과 적분의 차이가 뭐야?",
        "이차함수의 꼭짓점이 뭐야?",
        "극한(limit)이 뭔지 알려줘",
    ]
    cols_q = st.columns(3)
    for i, q in enumerate(quick_questions):
        if cols_q[i % 3].button(q, key=f"quick_{i}"):
            st.session_state.setdefault("pending_question", q)
            st.session_state.pending_question = q

    st.markdown("---")

    # 채팅 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 대화 기록 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 빠른 질문 처리
    pending = st.session_state.pop("pending_question", None)

    # 입력창
    user_input = st.chat_input("수학에 대해 무엇이든 물어보세요! 🎓")
    question = pending or user_input

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("AI 튜터가 생각하는 중... 🤔"):
                answer = ask_ollama(
                    question,
                    model_choice,
                    st.session_state.chat_history
                )
            st.markdown(f'<div class="ai-bubble">{answer}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # 사용법 안내
    with st.expander("💡 Ollama 설정 안내"):
        st.markdown("""
**Ollama 시작하기:**
```bash
# 1. Ollama 서버 시작
ollama serve

# 2. 모델 다운로드 (터미널에서)
ollama pull gemma4:e4b
ollama pull gemma4:26b

# 3. 모델 목록 확인
ollama list
```

**모델 안내:**
- `gemma4:e4b` : 빠른 응답, 가벼운 모델
- `gemma4:26b` : 더 풍부한 설명, 시간이 조금 더 걸려요
""")


# ══════════════════════════════════════════════════════════════════════════════
#  사이드바 + 라우터
# ══════════════════════════════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("""
<div style="text-align:center; padding:10px 0 20px;">
  <div style="font-size:2.5rem;">🧮</div>
  <div style="font-size:1.2rem; font-weight:900; color:#1a237e;">미적분 탐험대</div>
  <div style="font-size:.85rem; color:#888;">초등학생도 OK!</div>
</div>
""", unsafe_allow_html=True)

        selected = st.radio("챕터 선택", list(CHAPTERS.keys()), label_visibility="collapsed")

        st.markdown("---")
        st.markdown("#### 📊 학습 진도")
        chapters_done = ["📏 1. 수직선", "📊 2. 함숫값", "📐 3. 기울기",
                         "🎢 4. 이차함수", "⚡ 5. 미분은 기울기다", "🏞️ 6. 적분은 넓이다"]
        visited = st.session_state.get("visited_chapters", set())
        for ch in chapters_done:
            icon = "✅" if ch in visited else "⬜"
            st.markdown(f"{icon} {ch}")

        st.markdown("---")
        st.markdown("#### 🤖 AI 모델 상태")
        if OLLAMA_AVAILABLE:
            st.success("ollama 설치됨 ✓")
        else:
            st.error("ollama 미설치")

        return selected


def main():
    selected = sidebar()

    # 방문 기록
    if "visited_chapters" not in st.session_state:
        st.session_state.visited_chapters = set()
    if selected not in ("🏠 홈", "🤖 AI 수학 튜터"):
        st.session_state.visited_chapters.add(selected)

    page_key = CHAPTERS[selected]
    pages = {
        "home": page_home,
        "number_line": page_number_line,
        "function_value": page_function_value,
        "slope": page_slope,
        "quadratic": page_quadratic,
        "derivative": page_derivative,
        "integral": page_integral,
        "ai_tutor": page_ai_tutor,
    }
    pages[page_key]()


if __name__ == "__main__":
    main()
