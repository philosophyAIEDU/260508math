"""수포자도 이해하는 미적분 탐험대 — Streamlit + Ollama(gemma4)"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import io

# ── Ollama ──────────────────────────────────────────────────────────────────
try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ── 한글 폰트 ──────────────────────────────────────────────────────────────
def _set_korean_font():
    font_manager._load_fontmanager(try_read_cache=False)
    candidates = ["NanumGothic","NanumBarunGothic","NanumMyeongjo","Malgun Gothic","AppleGothic","Gulim"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    import glob
    paths = glob.glob("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
    if paths:
        font_manager.fontManager.addfont(paths[0])
        prop = font_manager.FontProperties(fname=paths[0])
        plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()

# ── 페이지 설정 ────────────────────────────────────────────────────────────
st.set_page_config(page_title="미적분 탐험대 🚀", page_icon="🧮",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans KR',sans-serif;}
.story-box{background:linear-gradient(135deg,#e8f4fd,#fce4ec);border-left:6px solid #2196F3;
  border-radius:12px;padding:20px 24px;margin:12px 0;font-size:1.05rem;line-height:1.9;color:#1a237e;}
.concept-card{background:#fff;border:2px solid #4CAF50;border-radius:14px;padding:18px;
  margin:10px 0;box-shadow:0 4px 12px rgba(76,175,80,.15);color:#333;}
.warn-card{background:#fff8e1;border:2px solid #FF9800;border-radius:14px;padding:16px;margin:10px 0;color:#333;}
.tip-card{background:#e8f5e9;border:2px solid #4CAF50;border-radius:14px;padding:16px;margin:10px 0;color:#1b5e20;}
.step-card{background:#f3e5f5;border:2px solid #9C27B0;border-radius:14px;padding:16px;margin:8px 0;color:#333;}
.highlight{background:#fff9c4;border-radius:6px;padding:2px 8px;font-weight:700;color:#e65100;}
.big-formula{background:#1a237e;color:#fff;border-radius:12px;padding:14px 20px;
  font-size:1.4rem;font-weight:700;text-align:center;margin:12px 0;}
.chapter-title{font-size:2rem;font-weight:900;color:#FFD700;margin-bottom:4px;}
.quiz-box{background:#e3f2fd;border:2px solid #1976D2;border-radius:12px;padding:16px;margin:10px 0;}
.ai-bubble{background:#e3f2fd;border-radius:18px 18px 18px 4px;padding:16px 20px;
  margin:10px 0;border-left:4px solid #1976D2;color:#1a237e;}
.summary-box{background:#fff3e0;border:2px solid #FF9800;border-radius:14px;
  padding:18px;margin:14px 0;color:#333;}
</style>
""", unsafe_allow_html=True)

# ── 챕터 정의 ──────────────────────────────────────────────────────────────
CHAPTERS = {
    "🏠 홈": "home",
    "📏 1. 수직선": "number_line",
    "📊 2. 함숫값": "function_value",
    "📐 3. 기울기": "slope",
    "🎢 4. 이차함수": "quadratic",
    "⚡ 5. 미분은 기울기다": "derivative",
    "🏞️ 6. 적분은 넓이다": "integral",
    "🧠 7. AI와 미적분": "ai_calculus",
    "🤖 AI 수학 튜터": "ai_tutor",
}

# ══════════════════════════════════════════════════════════════════════════════
#  그래프 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _show(fig):
    st.pyplot(fig); plt.close(fig)

def plot_number_line(highlight=None, op=None):
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(-7,7); ax.set_ylim(-1.0,1.0)
    ax.axhline(0, color="#333", lw=2.5)
    for x in range(-6,7):
        ax.plot(x,0,"o",color="#888",markersize=6)
        ax.text(x,-0.42,str(x),ha="center",va="top",fontsize=10,color="#333")
    for side,dx in [(6,-.6),(-6,.6)]:
        ax.annotate("",xy=(side,0),xytext=(side+dx,0),
                    arrowprops=dict(arrowstyle="->",color="#333",lw=2))
    if highlight is not None:
        ax.plot(highlight,0,"o",color="#F44336",markersize=14,zorder=5)
        ax.text(highlight,0.5,f"{highlight}",ha="center",fontsize=13,
                color="#F44336",fontweight="bold")
    if op:  # op = (start, end, label, color)
        s,e,lbl,col = op
        y_arc = 0.65
        ax.annotate("",xy=(e,0.05),xytext=(s,0.05),
                    arrowprops=dict(arrowstyle="->",color=col,lw=2.5,
                                    connectionstyle=f"arc3,rad={-0.4}"))
        ax.text((s+e)/2, y_arc, lbl, ha="center", fontsize=11, color=col, fontweight="bold")
        ax.plot(e,0,"D",color=col,markersize=11,zorder=6)
    ax.axis("off")
    ax.set_title("수직선 (Number Line)", fontsize=13, pad=6)
    return fig

def plot_function_map(pairs, title="함수 관계"):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.axis("off")
    n = len(pairs)
    for i,(x,y) in enumerate(pairs):
        yi = 0.15 + i*(0.7/(n-1)) if n>1 else 0.5
        ax.text(0.1, yi, f"x = {x}", ha="center", va="center", fontsize=14,
                bbox=dict(boxstyle="round,pad=0.4",fc="#BBDEFB",ec="#1976D2",lw=1.5))
        ax.annotate("", xy=(0.55, yi), xytext=(0.23, yi),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.8))
        ax.text(0.9, yi, f"y = {y}", ha="center", va="center", fontsize=14,
                bbox=dict(boxstyle="round,pad=0.4",fc="#C8E6C9",ec="#388E3C",lw=1.5))
    ax.text(0.38, 0.92, "f(x)", ha="center", fontsize=16, fontweight="bold",
            color="#9C27B0")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(title, fontsize=13)
    return fig

def plot_function_graph(func, x_vals, title="함수 그래프", color="#1976D2"):
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    y_vals = [func(x) for x in x_vals]
    # 입출력 표
    ax_t = axes[0]; ax_t.axis("off")
    t = ax_t.table(cellText=[[str(x), str(round(y,2))] for x,y in zip(x_vals,y_vals)],
                   colLabels=["x (입력)","y (출력)"], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1.4,1.9)
    for (r,c),cell in t.get_celld().items():
        if r==0: cell.set_facecolor("#1976D2"); cell.set_text_props(color="white",fontweight="bold")
        elif r%2==0: cell.set_facecolor("#e3f2fd")
    ax_t.set_title("입출력 표",fontsize=13,pad=6)
    # 그래프
    ax_g = axes[1]
    xs = np.linspace(min(x_vals)-.5,max(x_vals)+.5,300)
    ax_g.plot(xs,[func(x) for x in xs],color=color,lw=2.5)
    ax_g.scatter(x_vals,y_vals,color="#F44336",s=80,zorder=5)
    ax_g.axhline(0,color="#aaa",lw=0.8,linestyle="--")
    ax_g.axvline(0,color="#aaa",lw=0.8,linestyle="--")
    ax_g.set_title(title,fontsize=13); ax_g.grid(True,alpha=0.3)
    ax_g.set_xlabel("x"); ax_g.set_ylabel("y = f(x)")
    plt.tight_layout(); return fig

def plot_slope_visual(x1,y1,x2,y2):
    slope=(y2-y1)/(x2-x1); intercept=y1-slope*x1
    fig,ax=plt.subplots(figsize=(8,5))
    xs=np.linspace(x1-2,x2+2,200)
    ax.plot(xs,slope*xs+intercept,color="#1976D2",lw=2.8,label=f"기울기 = {slope:.2f}")
    ax.plot([x1,x2],[y1,y2],"ro",markersize=11)
    ax.annotate("",xy=(x2,y1),xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->",color="#4CAF50",lw=2.5))
    ax.annotate("",xy=(x2,y2),xytext=(x2,y1),
                arrowprops=dict(arrowstyle="->",color="#F44336",lw=2.5))
    ax.text((x1+x2)/2,y1-(abs(y2-y1)*0.15+0.4),f"가로: {x2-x1}",
            ha="center",fontsize=12,color="#4CAF50",fontweight="bold")
    ax.text(x2+0.2,(y1+y2)/2,f"세로: {y2-y1}",
            ha="left",fontsize=12,color="#F44336",fontweight="bold")
    ax.set_title(f"기울기 = {y2-y1} ÷ {x2-x1} = {slope:.2f}",fontsize=13)
    ax.legend(fontsize=11); ax.grid(True,alpha=0.3)
    ax.axhline(0,color="#aaa",lw=0.8); ax.axvline(0,color="#aaa",lw=0.8)
    return fig

def plot_slope_types():
    fig,axes=plt.subplots(1,4,figsize=(14,3.5))
    configs=[
        (2,"양의 기울기\n(오르막)","#4CAF50"),
        (-1.5,"음의 기울기\n(내리막)","#F44336"),
        (0,"기울기=0\n(평지)","#1976D2"),
        (None,"기울기=∞\n(수직선)","#9C27B0"),
    ]
    for ax,(m,title,col) in zip(axes,configs):
        x=np.linspace(-2,2,50)
        if m is None:
            ax.axvline(0,color=col,lw=3)
        else:
            ax.plot(x,m*x,color=col,lw=3)
        ax.axhline(0,color="#bbb",lw=0.8); ax.axvline(0,color="#bbb",lw=0.8)
        ax.set_xlim(-2.5,2.5); ax.set_ylim(-4,4)
        ax.set_title(title,fontsize=11,color=col,fontweight="bold")
        ax.grid(True,alpha=0.2); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); return fig

def plot_quadratic(a=1,b=0,c=0):
    x=np.linspace(-5,5,400); y=a*x**2+b*x+c
    vx=-b/(2*a); vy=a*vx**2+b*vx+c
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(x,y,color="#9C27B0",lw=2.8,label=f"y={a}x²+{b}x+{c}")
    ax.plot(vx,vy,"y*",markersize=20,zorder=5,label=f"꼭짓점({vx:.1f},{vy:.1f})")
    ax.axhline(0,color="#aaa",lw=0.8,linestyle="--")
    ax.axvline(vx,color="#FF9800",lw=1.5,linestyle=":",label=f"대칭축 x={vx:.1f}")
    ax.set_ylim(min(y)-1,max(y)+1)
    ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
    ax.set_title("이차함수 그래프",fontsize=13)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return fig

def plot_derivative_limit(x0=2.0):
    func=lambda x:x**2; dfunc=lambda x:2*x
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    x=np.linspace(-0.3,3.8,400)
    cols=plt.cm.Reds(np.linspace(0.25,0.95,5))
    h_vals=[2.0,1.0,0.5,0.2,0.05]
    for ax_idx,ax in enumerate(axes):
        ax.plot(x,[func(xi) for xi in x],color="#1976D2",lw=2.5,label="f(x)=x²",zorder=3)
        ax.plot(x0,func(x0),"ko",markersize=9,zorder=6)
        if ax_idx==0:
            for i,h in enumerate(h_vals):
                m=(func(x0+h)-func(x0))/h; b=func(x0)-m*x0
                xs=np.linspace(x0-.5,x0+h+.3,60)
                ax.plot(xs,m*xs+b,color=cols[i],lw=1.8,
                        label=f"h={h:.2f} → 기울기≈{m:.2f}",alpha=0.9)
                ax.plot(x0+h,func(x0+h),"o",color=cols[i],markersize=7)
            ax.set_title(f"h를 줄일수록 접선에 수렴 (x₀={x0})",fontsize=12)
            ax.legend(fontsize=8,loc="upper left")
        else:
            mt=dfunc(x0); bt=func(x0)-mt*x0
            xs_t=np.linspace(x0-1.8,x0+1.8,100)
            ax.plot(xs_t,mt*xs_t+bt,color="#F44336",lw=3,
                    label=f"접선: 기울기=f'({x0})={mt:.1f}",zorder=5)
            ax.set_title(f"극한값 = 미분값 f'({x0}) = {mt:.1f}",fontsize=12)
            ax.legend(fontsize=10)
        ax.set_xlim(-0.3,3.8); ax.set_ylim(-0.5,11)
        ax.grid(True,alpha=0.3)
        ax.axhline(0,color="#aaa",lw=0.8); ax.axvline(0,color="#aaa",lw=0.8)
        ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout(); return fig

def plot_deriv_app(func,dfunc,fn_label,df_label,x_range=(-3.5,3.5)):
    x_arr=np.linspace(*x_range,300)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.5))
    ax1.plot(x_arr,[func(xi) for xi in x_arr],color="#1976D2",lw=2.5,label=fn_label)
    ax2.plot(x_arr,[dfunc(xi) for xi in x_arr],color="#9C27B0",lw=2.5,label=df_label)
    for ax,title in [(ax1,"원래 함수 f(x)"),(ax2,"도함수 f'(x) — 기울기 함수")]:
        ax.axhline(0,color="#aaa",lw=0.8); ax.axvline(0,color="#aaa",lw=0.8)
        ax.grid(True,alpha=0.3); ax.set_xlabel("x"); ax.legend(fontsize=10)
        ax.set_title(title,fontsize=12)
    plt.tight_layout(); return fig

def plot_riemann(func,a,b,n,method="mid"):
    fig,ax=plt.subplots(figsize=(9,5))
    x=np.linspace(a-.5,b+.5,400); y=[func(xi) for xi in x]
    ax.plot(x,y,color="#1976D2",lw=2.5,zorder=4,label="f(x)")
    dx=(b-a)/n; total=0.0
    for xi in np.linspace(a,b-dx,n):
        h=func(xi) if method=="left" else (func(xi+dx) if method=="right" else func(xi+dx/2))
        total+=h*dx
        ax.add_patch(patches.Rectangle((xi,0),dx,h,lw=0.7,ec="#333",
                                        fc="#4CAF50",alpha=0.45))
    ax.axhline(0,color="#aaa",lw=0.8)
    ax.set_xlim(a-.5,b+.5); ax.set_ylim(0,max(y)*1.15)
    ax.set_title(f"리만 합 (n={n}개) ≈ {total:.4f}",fontsize=12)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=11); ax.grid(True,alpha=0.25)
    return fig

def plot_integral_area(func,a,b,exact,label=""):
    x_all=np.linspace(a-.5,b+.5,400)
    x_fill=np.linspace(a,b,300)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(x_all,[func(xi) for xi in x_all],color="#1976D2",lw=2.5,label="f(x)")
    ax.fill_between(x_fill,0,[func(xi) for xi in x_fill],
                    color="#FF9800",alpha=0.55,label=f"넓이={exact:.4f}")
    ax.axhline(0,color="#aaa",lw=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(label or f"정적분 넓이={exact:.4f}",fontsize=13)
    ax.legend(fontsize=11); ax.grid(True,alpha=0.25)
    return fig

# ── AI 튜터 ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 '미적분 탐험대 AI 튜터'입니다.
규칙:
1. 초등학생도 이해하도록 쉬운 말과 비유를 사용하세요.
2. 예시는 피자, 자전거, 자동차, 온도계 같은 일상 소재로 드세요.
3. 수식은 최대한 단순하게, 말로 풀어서 설명하세요.
4. 항상 응원의 말을 포함하세요.
5. 한국어로 답변하세요."""

def ask_ollama(question,model,history):
    if not OLLAMA_AVAILABLE:
        return "⚠️ ollama 패키지 없음. `pip install ollama` 실행 후 `ollama serve` 시작하세요."
    msgs=[{"role":"system","content":SYSTEM_PROMPT}]+history[-6:]+[{"role":"user","content":question}]
    try:
        r=_ollama.chat(model=model,messages=msgs)
        return r["message"]["content"]
    except Exception as e:
        err=str(e)
        if "not found" in err.lower() or "pull" in err.lower():
            return f"⚠️ 모델 '{model}' 없음. `ollama pull {model}` 실행하세요."
        return f"⚠️ Ollama 오류: {err}"

# ══════════════════════════════════════════════════════════════════════════════
#  퀴즈 헬퍼
# ══════════════════════════════════════════════════════════════════════════════
def quiz(key, question, options, answer_idx, explanation):
    """options: list[str], answer_idx: 0-based"""
    chosen = st.radio(question, options, key=f"q_{key}", index=None)
    if st.button("✅ 정답 확인", key=f"btn_{key}"):
        if chosen is None:
            st.warning("보기를 선택해주세요!")
        elif chosen == options[answer_idx]:
            st.success(f"🎉 정답! {explanation}")
        else:
            st.error(f"❌ 틀렸어요! 정답은 **'{options[answer_idx]}'** 예요.\n\n💡 {explanation}")

# ══════════════════════════════════════════════════════════════════════════════
#  홈
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
<div style="text-align:center;padding:28px 0;">
  <h1 style="font-size:3.2rem;color:#FFD700;text-shadow:2px 2px 6px rgba(0,0,0,.3);">
    🚀 미적분 탐험대</h1>
  <p style="font-size:1.3rem;color:#555;">수포자도 OK! 이야기로 배우는 미분·적분</p>
</div>""", unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    for col,(icon,title,desc) in zip([c1,c2,c3],[
        ("🎯","단계별 학습","수직선→함수→기울기→미분→적분, 순서대로!"),
        ("🎨","풍부한 시각화","모든 개념을 그래프로 눈으로 확인"),
        ("🤖","AI 튜터","gemma4가 모르는 개념 친절하게 설명"),
    ]):
        col.markdown(f"""<div class="concept-card" style="text-align:center;">
<div style="font-size:2.4rem;">{icon}</div>
<h3 style="color:#1976D2;">{title}</h3>
<p style="color:#555;">{desc}</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 미적분까지 가는 7단계 여정")
    steps=[("📏","수직선","숫자의 복도"),("📊","함숫값","마법 상자"),
           ("📐","기울기","얼마나 가파를까"),("🎢","이차함수","포물선의 세계"),
           ("⚡","미분","순간 기울기"),("🏞️","적분","곡선 아래 넓이"),("🧠","AI와 미적분","딥러닝의 심장")]
    cols=st.columns(7)
    for col,(icon,name,sub) in zip(cols,steps):
        col.markdown(f"""<div style="text-align:center;background:#f5f5f5;
border-radius:10px;padding:10px 4px;">
<div style="font-size:1.7rem;">{icon}</div>
<div style="font-weight:700;color:#1a237e;font-size:.88rem;">{name}</div>
<div style="font-size:.73rem;color:#777;">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 사이드바에서 챕터를 선택해 시작하세요! 1장부터 차례로 읽으면 미적분이 자연스럽게 이해돼요.")

# ══════════════════════════════════════════════════════════════════════════════
#  1장. 수직선
# ══════════════════════════════════════════════════════════════════════════════
def page_number_line():
    st.markdown('<div class="chapter-title">📏 1장. 수직선 — 모든 수학의 출발점</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🏫 <b>이야기: 숫자들이 사는 복도</b><br>
학교 복도를 상상해봐요. 복도 한가운데 "0번 교실"이 있고,<br>
오른쪽으로 갈수록 1번, 2번, 3번… 방이 있어요.<br>
왼쪽으로 가면 -1번, -2번, -3번… (지하 창고라고 생각해요!)<br>
이 복도가 바로 <span class="highlight">수직선</span>이에요!<br>
수직선 하나로 세상의 모든 숫자를 표현할 수 있어요. 🎉
</div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4=st.tabs(["📖 개념","🔍 단계별 이해","🎮 직접 해보기","🧩 퀴즈 6문제"])

    with tab1:
        st.markdown("""
<div class="concept-card">
<h4>✅ 수직선의 3가지 규칙</h4>
<ol>
<li><b>0이 중심:</b> 모든 수의 기준점이에요. 은행 잔고가 0이면 돈이 딱 없는 것처럼!</li>
<li><b>오른쪽 = 클수록:</b> 3은 2보다 오른쪽 → 3 > 2</li>
<li><b>왼쪽 = 작을수록:</b> -5는 -2보다 왼쪽 → -5 < -2</li>
</ol>
</div>""", unsafe_allow_html=True)

        _show(plot_number_line())

        col1,col2=st.columns(2)
        col1.markdown("""
<div class="concept-card">
<h4>📌 양수 vs 음수</h4>
<ul>
<li><b>양수(+):</b> 0보다 큰 수. 기온이 영상, 내 계좌에 돈이 있음</li>
<li><b>음수(-):</b> 0보다 작은 수. 기온이 영하, 내 계좌가 마이너스</li>
<li><b>0:</b> 양수도 음수도 아닌 특별한 기준점</li>
</ul>
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="concept-card">
<h4>📌 수직선 위의 덧셈·뺄셈</h4>
<ul>
<li><b>더하기(+) = 오른쪽으로 이동</b></li>
<li>예) 2 + 3 → 2에서 오른쪽으로 3칸 → 5</li>
<li><b>빼기(-) = 왼쪽으로 이동</b></li>
<li>예) 1 - 4 → 1에서 왼쪽으로 4칸 → -3</li>
</ul>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="tip-card">
<b>💡 절댓값 — 0까지의 거리</b><br>
|-3| = 3 (0에서 3칸 떨어져 있음)<br>
|5| = 5 (0에서 5칸 떨어져 있음)<br>
두 수의 거리 = |a - b|   예) 2와 -3의 거리 = |2-(-3)| = 5
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="summary-box">
<b>🔑 미적분과의 연결</b><br>
수직선은 x축과 y축의 기초예요. 함수 그래프, 미분, 적분 모두 수직선 두 개를 직각으로
붙인 <b>좌표평면</b> 위에서 이루어져요. 수직선을 제대로 이해하면 미적분의 절반은 끝난 거예요!
</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### 단계별로 수직선 익히기")

        st.markdown("#### ① 덧셈을 수직선으로 보기")
        c1,c2=st.columns(2)
        start_add=c1.number_input("출발 숫자",value=2,step=1,min_value=-5,max_value=5,key="add_s")
        move_add=c2.number_input("더할 수 (+오른쪽, -왼쪽)",value=3,step=1,min_value=-8,max_value=8,key="add_m")
        result_add=start_add+move_add
        if -6<=result_add<=6:
            _show(plot_number_line(highlight=result_add,
                op=(start_add,result_add,f"{start_add}+({move_add})={result_add}","#4CAF50")))
        st.success(f"📍 {start_add} + ({move_add}) = **{result_add}**")

        st.markdown("#### ② 두 수 사이의 거리 구하기")
        c1,c2=st.columns(2)
        a_dist=c1.number_input("수 A",value=-2,step=1,min_value=-5,max_value=5,key="da")
        b_dist=c2.number_input("수 B",value=4,step=1,min_value=-5,max_value=5,key="db")
        dist=abs(a_dist-b_dist)
        st.markdown(f"""
<div class="concept-card" style="text-align:center;font-size:1.2rem;">
|{a_dist} - {b_dist}| = |{a_dist-b_dist}| = <span class="highlight">{dist}</span>
&nbsp;&nbsp;→ 두 수 사이 거리는 <b>{dist}칸</b>
</div>""", unsafe_allow_html=True)

        st.markdown("#### ③ 부등호 방향 맞추기")
        st.markdown("수직선에서 **더 오른쪽에 있는 수**가 항상 더 큰 수예요!")
        c1,c2=st.columns(2)
        num_x=c1.number_input("왼쪽 수",value=-1,step=1,min_value=-10,max_value=10,key="bx")
        num_y=c2.number_input("오른쪽 수",value=3,step=1,min_value=-10,max_value=10,key="by")
        sym=">" if num_x>num_y else ("<" if num_x<num_y else "=")
        st.info(f"**{num_x} {sym} {num_y}** — {num_x}이 수직선에서 {'오른쪽' if num_x>num_y else '왼쪽' if num_x<num_y else '같은 위치'}에 있어요")

    with tab3:
        st.markdown("#### 🎮 내가 원하는 수 찾기")
        val=st.slider("수직선 위에 표시할 숫자",-6,6,0)
        _show(plot_number_line(highlight=val))
        msgs={True:"✅ 양수! 0보다 오른쪽, 기온으로 치면 영상이에요 ☀️",
              False:"❄️ 음수! 0보다 왼쪽, 기온으로 치면 영하예요",
              None:"🎯 0! 양수도 음수도 아닌 기준점이에요"}
        key = True if val>0 else (False if val<0 else None)
        st.info(msgs[key])

        st.markdown("#### 🌡️ 온도계로 보는 수직선")
        temp=st.slider("오늘 기온(°C)",-20,40,15)
        bar_color="#2196F3" if temp<0 else "#FF5722"
        st.markdown(f"""
<div style="background:#f5f5f5;border-radius:10px;padding:16px;text-align:center;">
<div style="font-size:2rem;">{'🥶' if temp<0 else '☀️' if temp>25 else '😊'}</div>
<div style="font-size:1.5rem;font-weight:700;color:{bar_color};">{temp}°C</div>
<div>수직선에서 <b>{'왼쪽(음수)' if temp<0 else '오른쪽(양수)'}</b> 구역이에요</div>
</div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown("### 🧩 수직선 퀴즈 6문제 — 모두 맞추면 1장 완료!")
        st.markdown("---")

        quiz("nl1","Q1. 수직선에서 가장 오른쪽에 있는 수는?",
             ["-10","0","7","-1"],2,
             "수직선에서 오른쪽일수록 큰 수예요. 7이 가장 커요!")

        st.markdown("---")
        quiz("nl2","Q2. -4 + 7을 수직선으로 계산하면?",
             ["-3","3","-11","11"],1,
             "-4에서 오른쪽으로 7칸 이동 → -4+7=3")

        st.markdown("---")
        quiz("nl3","Q3. |−8|의 값은?",
             ["-8","0","8","16"],2,
             "절댓값은 0으로부터의 거리예요. -8은 0에서 8칸 떨어져 있으니 8!")

        st.markdown("---")
        quiz("nl4","Q4. -3과 5 사이의 거리는?",
             ["2","8","3","5"],1,
             "|−3−5| = |−8| = 8. 수직선에서 -3부터 5까지 8칸!")

        st.markdown("---")
        quiz("nl5","Q5. 다음 중 틀린 부등호는?",
             ["-1 < 0","−5 < −2","3 > −10","−7 > −6"],3,
             "-7은 -6보다 왼쪽(더 작은 수)이에요. -7 < -6이 맞아요!")

        st.markdown("---")
        quiz("nl6","Q6. 수직선에서 0에서 오른쪽으로 4칸, 그 다음 왼쪽으로 9칸 이동하면?",
             ["5","-5","13","-13"],1,
             "0 + 4 = 4, 4 - 9 = -5. 왼쪽으로 9칸 이동!")

# ══════════════════════════════════════════════════════════════════════════════
#  2장. 함숫값
# ══════════════════════════════════════════════════════════════════════════════
def page_function_value():
    st.markdown('<div class="chapter-title">📊 2장. 함숫값 — 입력하면 출력이 나오는 마법</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🎰 <b>이야기: 마법 자판기</b><br>
학교 앞에 신기한 자판기가 있어요.<br>
"콜라" 버튼(입력 x)을 누르면 → 콜라(출력 y)가 나와요.<br>
"커피" 버튼을 누르면 → 커피가 나오죠.<br>
중요한 규칙: 같은 버튼을 누르면 <b>항상 같은 것</b>이 나와야 해요!<br>
콜라를 눌렀는데 오늘은 콜라, 내일은 사이다가 나오면? 고장난 자판기!<br>
수학의 <span class="highlight">함수</span>도 똑같아요: 입력 하나에 출력 <b>딱 하나</b>만! 📦✨
</div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4=st.tabs(["📖 개념","🔍 단계별 풀이","🎮 함수 탐험기","🧩 퀴즈 6문제"])

    with tab1:
        st.markdown("""
<div class="concept-card">
<h4>✅ 함수(Function)란?</h4>
<ul>
<li><b>정의:</b> 입력값(x) 하나에 출력값(y) 하나가 대응되는 관계</li>
<li><b>표기:</b> y = f(x) &nbsp;→&nbsp; "x를 f에 넣으면 y가 나온다"</li>
<li><b>핵심 조건:</b> 하나의 입력 → 반드시 하나의 출력 (두 개 이상이면 함수 아님!)</li>
</ul>
</div>""", unsafe_allow_html=True)

        col1,col2=st.columns(2)
        col1.markdown("""
<div class="tip-card">
<b>✅ 함수인 것</b>
<ul>
<li>자판기 (버튼→음료, 1대1)</li>
<li>y = x + 2 (3 → 5, 항상 같음)</li>
<li>y = x² (2 → 4, 항상 같음)</li>
</ul>
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="warn-card">
<b>❌ 함수 아닌 것</b>
<ul>
<li>고장난 자판기 (콜라 눌렀더니 커피도 나옴)</li>
<li>x = y² (예: x=4이면 y=2 또는 y=-2, 두 값!)</li>
</ul>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="concept-card">
<h4>📌 함수 표현 방법 3가지</h4>
<ol>
<li><b>수식:</b> y = 2x + 1 (규칙을 수식으로)</li>
<li><b>표(Table):</b> x와 y의 쌍을 나열</li>
<li><b>그래프:</b> 점들을 이어 곡선/직선으로</li>
</ol>
</div>""", unsafe_allow_html=True)

        _show(plot_function_map([(1,3),(2,5),(3,7),(4,9)],"y = 2x + 1 (입력→출력)"))

        st.markdown("""
<div class="concept-card">
<h4>📌 정의역 & 치역 — 넣는 것 vs 나오는 것</h4>
<ul>
<li><b>정의역(Domain):</b> 함수에 넣을 수 있는 x값 전체<br>
  예) 자판기에서 누를 수 있는 버튼들 전부</li>
<li><b>치역(Range):</b> 실제로 나오는 y값 전체<br>
  예) 자판기에서 실제로 나오는 음료 종류들</li>
</ul>
</div>

<div class="summary-box">
<b>🔑 미적분과의 연결</b><br>
미분은 함수 f(x)의 "순간 변화율"을 구하는 것, 적분은 함수 f(x) 아래 넓이를 구하는 것이에요.
함수를 모르면 미분·적분을 할 수 없어요! 함수는 미적분의 재료예요.
</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### 단계별 함수값 계산")

        st.markdown("""
<div class="step-card">
<b>📝 예제 1: f(x) = 3x - 1 에서 f(4) = ?</b><br>
① x 자리에 4를 대입: f(4) = 3×4 - 1<br>
② 곱하기 먼저: 3×4 = 12<br>
③ 빼기: 12 - 1 = <b>11</b><br>
✅ 정답: f(4) = 11
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="step-card">
<b>📝 예제 2: g(x) = x² + 2x 에서 g(3) = ?</b><br>
① x 자리에 3 대입: g(3) = 3² + 2×3<br>
② 제곱: 3² = 9<br>
③ 곱하기: 2×3 = 6<br>
④ 더하기: 9 + 6 = <b>15</b><br>
✅ 정답: g(3) = 15
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="step-card">
<b>📝 예제 3: 함수의 증가·감소 맛보기</b><br>
h(x) = 2x에서 x가 1씩 커질 때마다 y는 2씩 커져요.<br>
이 "x당 y의 변화량 = 2"가 바로 나중에 배울 <b>기울기(미분값)</b>예요!
</div>""", unsafe_allow_html=True)

        st.markdown("#### 🧮 직접 계산해보기")
        func_str=st.selectbox("함수 선택",["f(x) = 2x + 1","f(x) = x²","f(x) = 3x - 4","f(x) = x² - x + 1"])
        x_in=st.number_input("x 값 입력",value=3.0,step=0.5)
        func_eval={
            "f(x) = 2x + 1": lambda x: 2*x+1,
            "f(x) = x²": lambda x: x**2,
            "f(x) = 3x - 4": lambda x: 3*x-4,
            "f(x) = x² - x + 1": lambda x: x**2-x+1,
        }
        result=func_eval[func_str](x_in)
        st.markdown(f"""
<div class="big-formula">{func_str[5:]} 에서 x={x_in:.1f} 대입 → <span style="color:#FFD700;">{result:.2f}</span></div>
""", unsafe_allow_html=True)

    with tab3:
        st.markdown("#### 함수 그래프 그리기")
        f_choice=st.selectbox("그릴 함수",[
            "y = x","y = 2x","y = x + 3","y = x²","y = 2x + 1","y = -x + 4"])
        fmap={
            "y = x":(lambda x:x,"#1976D2"),
            "y = 2x":(lambda x:2*x,"#4CAF50"),
            "y = x + 3":(lambda x:x+3,"#FF9800"),
            "y = x²":(lambda x:x**2,"#9C27B0"),
            "y = 2x + 1":(lambda x:2*x+1,"#F44336"),
            "y = -x + 4":(lambda x:-x+4,"#00BCD4"),
        }
        fn,col=fmap[f_choice]
        _show(plot_function_graph(fn,list(range(-4,5)),f_choice,color=col))

        st.markdown("#### 🔢 함수값 표 만들기")
        st.dataframe({"x":list(range(-4,5)),
                      "y":[round(fn(x),2) for x in range(-4,5)]},
                     use_container_width=True)

    with tab4:
        st.markdown("### 🧩 함수 퀴즈 6문제")
        st.markdown("---")

        quiz("fv1","Q1. f(x) = 2x + 3 일 때 f(5) = ?",
             ["10","13","16","8"],1,
             "f(5) = 2×5 + 3 = 10 + 3 = 13")

        st.markdown("---")
        quiz("fv2","Q2. g(x) = x² 일 때 g(-3) = ?",
             ["-9","6","9","3"],2,
             "(-3)² = (-3)×(-3) = 9. 음수를 제곱하면 양수!")

        st.markdown("---")
        quiz("fv3","Q3. 다음 중 함수가 아닌 것은?",
             ["y = x + 1","y = x²","x = y² (y가 두 값 가능)","y = 3"],2,
             "x = y² 에서 예를 들어 x=4이면 y=2 또는 y=-2. 입력 하나에 출력이 두 개 → 함수 아님!")

        st.markdown("---")
        quiz("fv4","Q4. h(x) = 3x - 2 일 때 h(0) = ?",
             ["0","-2","1","3"],1,
             "h(0) = 3×0 - 2 = 0 - 2 = -2")

        st.markdown("---")
        quiz("fv5","Q5. y = x² 에서 x의 정의역이 {-2, 0, 3}일 때 치역은?",
             ["{-4, 0, 9}","{4, 0, 9}","{2, 0, 3}","{4, 0, -9}"],1,
             "(-2)²=4, 0²=0, 3²=9 → 치역은 {4, 0, 9}")

        st.markdown("---")
        quiz("fv6","Q6. f(x) = x² - 1 일 때, f(3) - f(2) = ?",
             ["4","5","3","1"],1,
             "f(3)=9-1=8, f(2)=4-1=3. 8-3=5. 이 차이가 바로 미분의 기초예요!")

# ══════════════════════════════════════════════════════════════════════════════
#  3장. 기울기
# ══════════════════════════════════════════════════════════════════════════════
def page_slope():
    st.markdown('<div class="chapter-title">📐 3장. 기울기 — 변화의 속도를 숫자로!</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🚗 <b>이야기: 내비게이션과 오르막길</b><br>
내비게이션이 "전방 500m 오르막길 15%" 라고 알려줘요.<br>
15%는 100m 앞으로 갈 때 15m 높아진다는 뜻이에요.<br>
이게 바로 <span class="highlight">기울기</span>예요 — <b>옆으로 간 거리 대비 위로 올라간 거리</b>!<br>
기울기가 클수록 더 가파른 언덕, 작을수록 완만해요. 🏔️<br>
나중에 미분을 배우면 "곡선의 기울기"도 구할 수 있게 돼요!
</div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4=st.tabs(["📖 개념","🔍 단계별 풀이","🎮 기울기 탐험","🧩 퀴즈 6문제"])

    with tab1:
        st.markdown("""
<div class="big-formula">기울기 = (y의 변화량) ÷ (x의 변화량) = (y₂−y₁) ÷ (x₂−x₁)</div>

<div class="concept-card">
<h4>✅ 기울기가 의미하는 것</h4>
<ul>
<li><b>"x가 1 증가할 때 y가 얼마나 변하냐?"</b></li>
<li>기울기=2 → 1칸 오른쪽 갈 때마다 2칸 위로</li>
<li>기울기=-3 → 1칸 오른쪽 갈 때마다 3칸 아래로</li>
<li>기울기=0 → 아무리 가도 높이 변화 없음 (평지)</li>
</ul>
</div>""", unsafe_allow_html=True)

        _show(plot_slope_types())

        col1,col2=st.columns(2)
        col1.markdown("""
<div class="tip-card">
<b>💡 속도도 기울기예요!</b><br>
자동차가 3시간 동안 180km 이동했다면:<br>
속도 = 180 ÷ 3 = 60 km/h<br>
이게 바로 '거리-시간 그래프의 기울기'예요!<br>
<b>미분</b>은 이 기울기를 곡선에서도 구해요.
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="warn-card">
<b>⚠️ 주의! 수직선의 기울기</b><br>
수직으로 선 직선(x = 상수)은<br>
분모(x 변화량) = 0이라 기울기가 존재하지 않아요.<br>
0으로 나누기는 수학에서 금지!
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="summary-box">
<b>🔑 미적분과의 연결</b><br>
• 직선의 기울기는 항상 일정해요 (예: y=2x+1 → 기울기 항상 2)<br>
• 곡선의 기울기는 위치마다 달라요 (x=1일 때와 x=3일 때 다름)<br>
• <b>미분</b>은 바로 이 "곡선 위 특정 점에서의 기울기"를 구하는 방법이에요!
</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### 기울기 단계별 계산")

        st.markdown("""
<div class="step-card">
<b>📝 예제 1: 점 (1, 2)와 (4, 8)을 지나는 직선의 기울기</b><br>
① y의 변화량: 8 - 2 = 6<br>
② x의 변화량: 4 - 1 = 3<br>
③ 기울기 = 6 ÷ 3 = <b>2</b><br>
해석: x가 1 증가할 때마다 y가 2씩 증가해요
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="step-card">
<b>📝 예제 2: 점 (0, 5)와 (3, -1)을 지나는 직선의 기울기</b><br>
① y의 변화량: -1 - 5 = -6<br>
② x의 변화량: 3 - 0 = 3<br>
③ 기울기 = -6 ÷ 3 = <b>-2</b><br>
해석: x가 1 증가할 때마다 y가 2씩 <b>감소</b>해요 (내리막!)
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="step-card">
<b>📝 예제 3: 일차함수 y = 3x + 7 의 기울기는?</b><br>
y = (기울기)x + (y절편) 형태에서<br>
기울기 = x 앞의 수 = <b>3</b><br>
y절편 (x=0일 때 y값) = 7
</div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("#### 🎮 두 점으로 기울기 직접 계산!")
        c1,c2=st.columns(2)
        x1=c1.number_input("점 A의 x",value=1.0,step=0.5,key="sx1")
        y1=c1.number_input("점 A의 y",value=2.0,step=0.5,key="sy1")
        x2=c2.number_input("점 B의 x",value=4.0,step=0.5,key="sx2")
        y2=c2.number_input("점 B의 y",value=8.0,step=0.5,key="sy2")

        if abs(x2-x1)<0.001:
            st.error("⚠️ x값이 같으면 기울기가 존재하지 않아요! (수직선)")
        else:
            slope=(y2-y1)/(x2-x1)
            st.markdown(f"""
<div class="big-formula">기울기 = ({y2:.1f} - {y1:.1f}) ÷ ({x2:.1f} - {x1:.1f}) =
<span style="color:#FFD700;">{slope:.2f}</span></div>""", unsafe_allow_html=True)
            _show(plot_slope_visual(x1,y1,x2,y2))
            msg=("🏔️ 가파른 오르막이에요!" if slope>3 else
                 "📈 오르막이에요." if slope>0 else
                 "📉 내리막이에요." if slope<0 else "➡️ 완전 평지예요!")
            st.info(msg)

        st.markdown("---")
        st.markdown("#### 일차함수 y = ax + b 그래프")
        ca,cb=st.columns(2)
        a_sl=ca.slider("기울기 a",-5.0,5.0,2.0,0.5)
        b_sl=cb.slider("y절편 b (x=0일 때 y)",-5.0,5.0,1.0,0.5)
        _show(plot_function_graph(lambda x:a_sl*x+b_sl,list(range(-4,5)),
                                   f"y = {a_sl}x + {b_sl}",color="#1976D2"))
        st.info(f"기울기={a_sl} | x가 1 증가할 때 y는 {a_sl:+.1f} 변해요")

    with tab4:
        st.markdown("### 🧩 기울기 퀴즈 6문제")
        st.markdown("---")

        quiz("sl1","Q1. 점 (2,4)와 (6,12)를 지나는 직선의 기울기는?",
             ["2","3","4","6"],0,
             "(12-4)÷(6-2) = 8÷4 = 2")

        st.markdown("---")
        quiz("sl2","Q2. y = -4x + 10 의 기울기는?",
             ["10","-4","4","-10"],1,
             "y=ax+b 에서 기울기는 x 앞 계수 a = -4")

        st.markdown("---")
        quiz("sl3","Q3. 기울기가 0인 직선의 모양은?",
             ["수직으로 서 있는 선","오른쪽 위로 올라가는 선",
              "수평으로 평평한 선","오른쪽 아래로 내려가는 선"],2,
             "기울기=0 → 변화 없음 → 수평(평평한) 직선!")

        st.markdown("---")
        quiz("sl4","Q4. 자동차가 2시간 동안 120km 이동했어요. 이 '속도'는 어떤 개념과 같을까요?",
             ["y절편","기울기","정의역","치역"],1,
             "속도 = 거리÷시간 = 120÷2 = 60 km/h. 이것은 '거리-시간 그래프의 기울기'예요!")

        st.markdown("---")
        quiz("sl5","Q5. 점 (0,0)과 (5,−10)을 지나는 직선의 기울기는?",
             ["2","-2","5","-10"],1,
             "(-10-0)÷(5-0) = -10÷5 = -2. 내리막 방향!")

        st.markdown("---")
        quiz("sl6","Q6. 기울기가 클수록 그래프는?",
             ["더 완만해진다","더 가파르게 올라간다","더 가파르게 내려간다","변화 없다"],1,
             "기울기가 클수록 같은 x 이동에서 y 변화가 커지니 더 가파르게 올라가요!")

# ══════════════════════════════════════════════════════════════════════════════
#  4장. 이차함수
# ══════════════════════════════════════════════════════════════════════════════
def page_quadratic():
    st.markdown('<div class="chapter-title">🎢 4장. 이차함수 — 포물선의 세계</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
⚽ <b>이야기: 공을 차면 왜 포물선을 그릴까?</b><br>
축구공을 발로 힘껏 차보세요. 공은 처음엔 올라가다가 어느 순간 최고점에 도달하고,<br>
그 다음은 내려와요. 이 궤적을 그려보면 정확히 <b>포물선</b> 모양이에요!<br>
이 아름다운 곡선을 수식으로 표현하면 <span class="highlight">y = ax² + bx + c</span><br>
이것이 바로 <b>이차함수</b>예요. 자연 속 수많은 현상이 이 공식을 따라요! 🌍
</div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4=st.tabs(["📖 개념","🔍 단계별 분석","🎮 이차함수 조각가","🧩 퀴즈 6문제"])

    with tab1:
        col1,col2=st.columns(2)
        col1.markdown("""
<div class="concept-card">
<h4>✅ y = ax² + bx + c</h4>
<ul>
<li><b>a (가장 중요!):</b><br>
  a > 0 → ∪ 모양 (오목, 아래가 최솟값)<br>
  a < 0 → ∩ 모양 (볼록, 위가 최댓값)<br>
  |a|가 클수록 포물선이 더 좁아요</li>
<li><b>c:</b> y절편. x=0일 때 y값</li>
</ul>
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="concept-card">
<h4>✅ 꼭짓점과 대칭축</h4>
<ul>
<li><b>꼭짓점:</b> 포물선의 최고점 또는 최저점<br>
  x좌표 = -b/(2a)</li>
<li><b>대칭축:</b> 꼭짓점을 지나는 수직선<br>
  x = -b/(2a)</li>
<li>포물선은 대칭축을 기준으로 좌우 대칭!</li>
</ul>
</div>""", unsafe_allow_html=True)

        _show(plot_quadratic(1,0,0))

        st.markdown("""
<div class="step-card">
<b>📝 꼭짓점 구하는 공식 기억법</b><br>
y = ax² + bx + c 에서 꼭짓점의 x좌표 = <b>-b ÷ (2a)</b><br>
예) y = x² - 4x + 3 → a=1, b=-4<br>
꼭짓점 x = -(-4) ÷ (2×1) = 4 ÷ 2 = <b>2</b><br>
꼭짓점 y = 2² - 4×2 + 3 = 4 - 8 + 3 = <b>-1</b><br>
꼭짓점: (2, -1)
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="summary-box">
<b>🔑 미적분과의 연결</b><br>
이차함수의 꼭짓점에서 기울기(미분값)는 0이에요!<br>
나중에 배울 미분을 이용하면 f'(x)=0 되는 점을 찾아서 꼭짓점을 구할 수 있어요.<br>
이차함수는 미분 연습에 가장 좋은 함수예요.
</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### 이차함수 단계별 분석")

        st.markdown("""
<div class="step-card">
<b>📝 예제: y = 2x² - 8x + 6 분석하기</b><br>
① a=2 > 0 이므로 ∪ 모양 (아래로 볼록)<br>
② 꼭짓점 x = -(-8)÷(2×2) = 8÷4 = 2<br>
③ 꼭짓점 y = 2×4 - 8×2 + 6 = 8 - 16 + 6 = -2<br>
④ 꼭짓점: (2, -2) → 이 점이 최솟값!<br>
⑤ y절편(x=0): y = 0 - 0 + 6 = 6
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="step-card">
<b>📝 실생활 예제: 공 던지기</b><br>
공을 위로 던졌을 때 높이: h(t) = -5t² + 20t (단위: m, 초)<br>
① 최고점 시간: t = -20÷(2×(-5)) = 20÷10 = 2초<br>
② 최고 높이: h(2) = -5×4 + 20×2 = -20+40 = 20m<br>
③ 땅에 떨어지는 시간: h(t)=0 → t=0 또는 t=4초
</div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("#### 🎮 a, b, c 직접 조절하기!")
        c1,c2,c3=st.columns(3)
        a=c1.slider("a (포물선 방향/폭)",-3.0,3.0,1.0,0.5)
        b=c2.slider("b (좌우 이동 영향)",-6.0,6.0,0.0,0.5)
        c=c3.slider("c (위아래 이동)",-5.0,5.0,0.0,0.5)

        if abs(a)<0.01:
            st.warning("a=0이면 일차함수예요! 이차함수가 되려면 a≠0이어야 해요.")
        else:
            vx=-b/(2*a); vy=a*vx**2+b*vx+c
            st.markdown(f"""
<div class="concept-card" style="text-align:center;">
y = {a}x² {'+' if b>=0 else ''}{b}x {'+' if c>=0 else ''}{c}
&nbsp;&nbsp;|&nbsp;&nbsp; 꼭짓점: ({vx:.2f}, {vy:.2f})
&nbsp;&nbsp;|&nbsp;&nbsp; {'∪ 아래로 볼록 (최솟값)' if a>0 else '∩ 위로 볼록 (최댓값)'}
</div>""", unsafe_allow_html=True)
            _show(plot_quadratic(a,b,c))

    with tab4:
        st.markdown("### 🧩 이차함수 퀴즈 6문제")
        st.markdown("---")

        quiz("qd1","Q1. y = x² - 6x + 8 의 꼭짓점 x좌표는?",
             ["3","6","8","−3"],0,
             "x = -b/(2a) = -(-6)/(2×1) = 6/2 = 3")

        st.markdown("---")
        quiz("qd2","Q2. y = -2x² + 4x - 1 에서 포물선의 방향은?",
             ["위로 열림 ∪","아래로 열림 ∩","수평","수직"],1,
             "a = -2 < 0 이므로 ∩ 모양 (아래로 열림)")

        st.markdown("---")
        quiz("qd3","Q3. y = 3x² + 0x - 5 의 y절편(x=0일 때)은?",
             ["3","0","-5","5"],2,
             "x=0 대입: y = 3×0 - 0 - 5 = -5")

        st.markdown("---")
        quiz("qd4","Q4. 공을 던졌을 때 높이 h(t) = -t² + 6t 에서 최고점에 도달하는 시간은?",
             ["1초","3초","6초","9초"],1,
             "t = -b/(2a) = -6/(2×(-1)) = 6/2 = 3초")

        st.markdown("---")
        quiz("qd5","Q5. y = x² 와 비교해서 y = 5x² 는?",
             ["더 넓고 완만한 포물선","더 좁고 가파른 포물선",
              "아래로 뒤집힌 포물선","같은 모양"],1,
             "|a|=5 > 1이므로 기본 포물선보다 더 좁고 가파르게 솟아요")

        st.markdown("---")
        quiz("qd6","Q6. 이차함수 꼭짓점에서 기울기(미분값)는?",
             ["1","-1","0","2"],2,
             "꼭짓점은 최고점 또는 최저점 → 기울기 변화 없음 → 미분값=0! 미분을 배우면 이걸 증명할 수 있어요!")

# ══════════════════════════════════════════════════════════════════════════════
#  5장. 미분
# ══════════════════════════════════════════════════════════════════════════════
def page_derivative():
    st.markdown('<div class="chapter-title">⚡ 5장. 미분은 기울기다 — 순간의 속도를 잡아라!</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🏎️ <b>이야기: 자동차 속도계의 비밀</b><br>
자동차를 타면 속도계가 "현재 60km/h"를 가리켜요.<br>
그런데 이 60km/h는 어떻게 측정한 걸까요?<br>
"지금 이 순간" 속도를 측정하려면, 아주 짧은 시간(0.001초) 동안 이동한 거리를 재야 해요.<br>
시간을 더 짧게, 더 짧게, 0에 가깝게 → 진짜 <b>순간 속도</b>!<br>
이 과정이 바로 <span class="highlight">미분(극한)</span>이에요. 수학의 가장 위대한 발명! 🚀
</div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4,tab5=st.tabs(["📖 개념","🔬 극한 이해","🎮 미분 탐험","📐 미분 공식","🧩 퀴즈 6문제"])

    with tab1:
        st.markdown("""
<div class="concept-card">
<h4>✅ 미분의 핵심 아이디어</h4>
<ol>
<li>곡선 위에 점 A와 점 B를 찍어요</li>
<li>두 점을 이은 직선(할선)의 기울기를 구해요</li>
<li>B를 A 쪽으로 점점 당겨요 (h → 0)</li>
<li>두 점이 거의 붙으면, 그 기울기가 <b>접선의 기울기</b></li>
<li>접선의 기울기 = <b>미분값 f'(x)</b></li>
</ol>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="big-formula">f'(x) = lim [f(x+h) - f(x)] / h &nbsp; (h→0)</div>""",
                    unsafe_allow_html=True)

        col1,col2=st.columns(2)
        col1.markdown("""
<div class="tip-card">
<b>💡 극한이 뭔가요?</b><br>
극한(limit)은 "한없이 가까이 가면 얼마가 되는가?"예요.<br>
예) h를 점점 작게 하면:<br>
h=1 → 기울기≈3<br>
h=0.1 → 기울기≈2.1<br>
h=0.01 → 기울기≈2.01<br>
h→0 → 기울기→<b>2</b> (이게 미분값!)
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="concept-card">
<b>📌 미분값이 말해주는 것</b><br>
<ul>
<li>f'(x) > 0 → 그 점에서 함수가 증가 중 📈</li>
<li>f'(x) < 0 → 그 점에서 함수가 감소 중 📉</li>
<li>f'(x) = 0 → 꼭짓점! (최대·최소) ⭐</li>
</ul>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="summary-box">
<b>🔑 미분과 기울기의 관계 정리</b><br>
• 직선: 기울기가 어디서나 일정 (이미 배운 기울기!)<br>
• 곡선: 위치마다 기울기가 달라요<br>
• <b>미분 = 곡선의 각 점에서 기울기를 구하는 방법</b><br>
• f(x)=x² 이면 x=1에서 기울기=2, x=3에서 기울기=6
</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 🔬 h를 줄여가며 극한 직접 체험!")
        st.write("f(x) = x² 에서 주어진 점의 순간 기울기(미분값)를 극한으로 구해봐요")

        x0v=st.slider("x₀ 위치 선택",0.5,3.0,2.0,0.5)
        hv=st.select_slider("h 값 (점점 0에 가깝게!)",
                             options=[2.0,1.0,0.5,0.2,0.1,0.01,0.001,0.0001],value=1.0)
        f=lambda x:x**2
        approx=(f(x0v+hv)-f(x0v))/hv
        true_val=2*x0v

        col1,col2,col3=st.columns(3)
        col1.metric("h 값",f"{hv}")
        col2.metric("근사 기울기",f"{approx:.6f}")
        col3.metric("진짜 미분값",f"{true_val:.4f}",delta=f"오차: {abs(approx-true_val):.6f}")

        if abs(approx-true_val)<0.01:
            st.success(f"🎉 거의 완벽해요! h={hv}로 미분값 {true_val}에 수렴했어요!")
        else:
            st.info(f"💡 h를 더 작게 줄여봐요. 미분값 {true_val}에 점점 가까워지고 있어요!")

        _show(plot_derivative_limit(x0v))

    with tab3:
        st.markdown("#### 🎮 함수별 미분 탐험기")
        fn_ch=st.selectbox("함수 선택",[
            "f(x) = x²","f(x) = x³","f(x) = 3x² + 2x","f(x) = x² - 4x + 3"])
        fn_map={
            "f(x) = x²": (lambda x:x**2, lambda x:2*x, "f'(x) = 2x"),
            "f(x) = x³": (lambda x:x**3, lambda x:3*x**2, "f'(x) = 3x²"),
            "f(x) = 3x² + 2x": (lambda x:3*x**2+2*x, lambda x:6*x+2, "f'(x) = 6x + 2"),
            "f(x) = x² - 4x + 3": (lambda x:x**2-4*x+3, lambda x:2*x-4, "f'(x) = 2x - 4"),
        }
        fn_v,df_v,df_label=fn_map[fn_ch]
        x_pt=st.slider("점 x 선택",-3.0,3.0,1.0,0.5)

        col1,col2=st.columns(2)
        col1.metric(f"함수값 f({x_pt})",f"{fn_v(x_pt):.3f}")
        col2.metric(f"미분값 f'({x_pt}) = 순간 기울기",f"{df_v(x_pt):.3f}")
        st.markdown(f"**도함수:** {df_label}")

        _show(plot_deriv_app(fn_v,df_v,fn_ch,df_label))

        st.markdown("""
<div class="tip-card">
<b>💡 도함수 읽는 법</b><br>
오른쪽 그래프(도함수)에서:<br>
• 도함수 > 0인 구간 → 원래 함수가 증가<br>
• 도함수 < 0인 구간 → 원래 함수가 감소<br>
• 도함수 = 0인 점 → 꼭짓점 (최대 또는 최소)
</div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown("""
<div class="big-formula">기본 미분 공식: (xⁿ)' = n·xⁿ⁻¹</div>

<div class="concept-card">
<h4>📐 자주 쓰는 미분 공식 표</h4>
<table style="width:100%;font-size:1.05rem;border-collapse:collapse;">
<tr style="background:#1976D2;color:white;">
<th style="padding:8px;">함수 f(x)</th><th>미분 f'(x)</th><th>쉬운 설명</th></tr>
<tr style="background:#e3f2fd;"><td style="padding:6px;text-align:center;">상수 (예: 5)</td>
<td style="text-align:center;">0</td><td>상수는 변화 없음 → 기울기 0</td></tr>
<tr><td style="padding:6px;text-align:center;">x</td>
<td style="text-align:center;">1</td><td>y=x는 기울기가 항상 1</td></tr>
<tr style="background:#e3f2fd;"><td style="padding:6px;text-align:center;">x²</td>
<td style="text-align:center;">2x</td><td>지수 2를 앞으로, 지수는 1 감소</td></tr>
<tr><td style="padding:6px;text-align:center;">x³</td>
<td style="text-align:center;">3x²</td><td>지수 3을 앞으로, 지수는 2</td></tr>
<tr style="background:#e3f2fd;"><td style="padding:6px;text-align:center;">xⁿ</td>
<td style="text-align:center;">n·xⁿ⁻¹</td><td>지수를 앞으로 끌어내고, 지수 1 감소</td></tr>
<tr><td style="padding:6px;text-align:center;">3x² + 2x</td>
<td style="text-align:center;">6x + 2</td><td>각 항을 따로따로 미분!</td></tr>
</table>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="step-card">
<b>📝 연습: f(x) = 4x³ - 3x² + 2x - 7 미분하기</b><br>
각 항을 따로 미분하면:<br>
• 4x³ → 4×3·x² = 12x²<br>
• -3x² → -3×2·x = -6x<br>
• 2x → 2×1 = 2<br>
• -7 → 0 (상수 미분 = 0)<br>
∴ f'(x) = 12x² - 6x + 2
</div>""", unsafe_allow_html=True)

    with tab5:
        st.markdown("### 🧩 미분 퀴즈 6문제")
        st.markdown("---")

        quiz("dv1","Q1. f(x) = x² 일 때 f'(3) = ?",
             ["3","6","9","12"],1,
             "f'(x) = 2x, f'(3) = 2×3 = 6. x=3에서 접선 기울기는 6!")

        st.markdown("---")
        quiz("dv2","Q2. f(x) = 5 (상수함수)의 미분값은?",
             ["5","1","0","-5"],2,
             "상수는 변하지 않으므로 기울기=0. 평평한 직선의 기울기!")

        st.markdown("---")
        quiz("dv3","Q3. f(x) = x³ 의 도함수 f'(x) = ?",
             ["x²","3x","3x²","x²/3"],2,
             "xⁿ → nxⁿ⁻¹ 공식: x³ → 3x²")

        st.markdown("---")
        quiz("dv4","Q4. f(x) = 2x² - 4x 에서 f'(x) = 0이 되는 x는?",
             ["x=0","x=1","x=2","x=4"],1,
             "f'(x)=4x-4=0 → x=1. 이 점이 꼭짓점(최솟값)!")

        st.markdown("---")
        quiz("dv5","Q5. 미분값 f'(x) > 0 이면?",
             ["함수가 감소하는 구간","함수가 최솟값","함수가 증가하는 구간","기울기가 0"],2,
             "f'(x)>0 → 접선이 위로 올라가는 방향 → 함수 증가!")

        st.markdown("---")
        quiz("dv6","Q6. h(x) = 6x³ - 2x 의 도함수 h'(x) = ?",
             ["6x²-2","18x²-2","18x³-2","6x²"],1,
             "6x³ → 6×3x² = 18x², -2x → -2. 합치면 18x²-2")

# ══════════════════════════════════════════════════════════════════════════════
#  6장. 적분
# ══════════════════════════════════════════════════════════════════════════════
def page_integral():
    st.markdown('<div class="chapter-title">🏞️ 6장. 적분은 넓이다 — 잘게 쪼개면 보인다!</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🌾 <b>이야기: 구불구불한 밭의 넓이 구하기</b><br>
농부 아저씨에게 구불구불한 모양의 밭이 있어요. 이 밭의 넓이를 어떻게 구할까요?<br>
방법: 밭을 아주 얇은 <b>직사각형 막대</b>들로 채워봐요!<br>
막대가 100개 → 그럭저럭 맞음<br>
막대가 1000개 → 더 정확!<br>
막대가 무한개 → 완벽하게 정확! 이것이 <span class="highlight">적분</span>이에요! 🎯
</div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4,tab5=st.tabs(["📖 개념","🔬 리만 합","🎮 정적분 계산","🌍 응용","🧩 퀴즈 6문제"])

    with tab1:
        col1,col2=st.columns(2)
        col1.markdown("""
<div class="concept-card">
<h4>✅ 적분의 핵심 아이디어</h4>
<ol>
<li>구간 [a,b]를 n개 조각으로 나눠요</li>
<li>각 조각에서 직사각형 넓이 = f(x)×Δx</li>
<li>모든 직사각형 넓이의 합 = 리만 합</li>
<li>n→∞ (무한히 나누면) = 정적분!</li>
</ol>
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="big-formula">∫ₐᵇ f(x) dx<br>
<span style="font-size:1rem;font-weight:400;">a: 시작, b: 끝, f(x): 높이, dx: 무한히 얇은 폭</span>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="concept-card">
<h4>📌 미적분학의 기본 정리 (Fundamental Theorem of Calculus)</h4>
<p>미분과 적분은 서로 <b>역연산</b>이에요!</p>
<ul>
<li>F(x)를 미분하면 f(x)</li>
<li>f(x)를 적분하면 F(x)</li>
<li>∫ₐᵇ f(x)dx = F(b) - F(a) &nbsp; (여기서 F'(x) = f(x))</li>
</ul>
<p>마치 덧셈↔뺄셈, 곱셈↔나눗셈처럼!</p>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="step-card">
<b>📝 예제: ∫₀³ x² dx 계산하기</b><br>
① x²를 적분하면 → x³/3 (미분의 역!)<br>
② F(x) = x³/3<br>
③ F(3) - F(0) = 3³/3 - 0 = 27/3 - 0 = <b>9</b><br>
✅ x=0부터 x=3까지 x² 아래 넓이 = 9
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="summary-box">
<b>🔑 자주 쓰는 적분 공식</b><br>
• ∫xⁿ dx = xⁿ⁺¹/(n+1) + C &nbsp; (n≠-1)<br>
• ∫x² dx = x³/3 + C<br>
• ∫2x dx = x² + C<br>
• ∫상수k dx = kx + C<br>
(C는 적분 상수)
</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 🔬 리만 합 — 직사각형으로 넓이 근사하기")
        fn3c=st.selectbox("함수 선택",["f(x)=x²","f(x)=x³","f(x)=2x+1","f(x)=x²+1"],key="rfn")
        fn3m={"f(x)=x²":lambda x:x**2,"f(x)=x³":lambda x:x**3,
              "f(x)=2x+1":lambda x:2*x+1,"f(x)=x²+1":lambda x:x**2+1}
        fn3=fn3m[fn3c]

        c1r,c2r,c3r=st.columns(3)
        ar=c1r.number_input("시작 a",value=0.0,step=0.5,key="ra")
        br=c2r.number_input("끝 b",value=3.0,step=0.5,key="rb")
        nr=c3r.slider("직사각형 수 n",1,200,10)

        if ar>=br:
            st.error("b > a 이어야 해요!")
        else:
            met=st.radio("높이 기준",["mid(중간)","left(왼쪽)","right(오른쪽)"],horizontal=True)
            _show(plot_riemann(fn3,ar,br,nr,method=met.split("(")[0]))
            # 정확한 값과 비교
            from scipy import integrate as sci_int
            exact,_=sci_int.quad(fn3,ar,br)
            dx=(br-ar)/nr
            approx=sum((fn3(ar+dx*(i+0.5)) if "mid" in met else
                        (fn3(ar+dx*i) if "left" in met else fn3(ar+dx*(i+1))))
                       *dx for i in range(nr))
            col1,col2,col3=st.columns(3)
            col1.metric("직사각형 수",f"{nr}개")
            col2.metric("리만 합 (근사)",f"{approx:.4f}")
            col3.metric("정확한 넓이",f"{exact:.4f}",delta=f"오차:{abs(approx-exact):.4f}")
            st.info(f"💡 n을 크게 할수록 오차가 줄어들어요! n={nr}일 때 오차: {abs(approx-exact):.4f}")

    with tab3:
        st.markdown("#### 🎮 정적분 계산기")
        fn4c=st.selectbox("함수",["f(x)=x²","f(x)=2x","f(x)=x²+1","f(x)=x³","f(x)=3x²-2"],key="ifn")
        fn4m={
            "f(x)=x²": (lambda x:x**2,"x³/3",lambda a,b:(b**3-a**3)/3),
            "f(x)=2x": (lambda x:2*x,"x²",lambda a,b:b**2-a**2),
            "f(x)=x²+1": (lambda x:x**2+1,"x³/3+x",lambda a,b:(b**3-a**3)/3+(b-a)),
            "f(x)=x³": (lambda x:x**3,"x⁴/4",lambda a,b:(b**4-a**4)/4),
            "f(x)=3x²-2": (lambda x:3*x**2-2,"x³-2x",lambda a,b:(b**3-a**3)-2*(b-a)),
        }
        fn4,anti,exact_fn=fn4m[fn4c]
        c1i,c2i=st.columns(2)
        ai=c1i.number_input("a",value=0.0,step=0.5,key="ia")
        bi=c2i.number_input("b",value=2.0,step=0.5,key="ib")
        if ai>=bi:
            st.error("b > a 이어야 해요!")
        else:
            area=exact_fn(ai,bi)
            st.markdown(f"""
<div class="big-formula">∫<sub>{ai}</sub><sup>{bi}</sup> {fn4c[5:]} dx = [{anti}]<sub>{ai}</sub><sup>{bi}</sup>
= <span style="color:#FFD700;">{area:.4f}</span></div>""", unsafe_allow_html=True)
            _show(plot_integral_area(fn4,ai,bi,area,f"∫ {fn4c[5:]} dx = {area:.4f}"))

    with tab4:
        st.markdown("#### 🌍 적분의 실생활 응용")
        apps=[
            ("🚗 이동 거리","속도 함수를 적분하면 총 이동 거리!\n예) v(t) = 60 → ∫₀²60 dt = 120 km","#e3f2fd"),
            ("💧 물의 총량","유량 함수를 적분하면 총 물의 양!\n수도꼭지 1시간 동안 흘러나온 물","#e8f5e9"),
            ("🌡️ 평균 온도","온도 함수를 적분 후 시간으로 나누면 평균 온도","#fff8e1"),
            ("🔋 전기 에너지","전력(와트)을 시간에 대해 적분하면 에너지(Wh)","#fce4ec"),
            ("📊 통계·확률","확률밀도함수를 적분하면 확률!\nAI, 데이터 분석의 핵심","#f3e5f5"),
            ("🏗️ 구조물 무게","밀도 함수를 적분하면 전체 질량!","#e0f2f1"),
        ]
        cols=st.columns(2)
        for i,(icon,desc,bg) in enumerate(apps):
            cols[i%2].markdown(f"""<div class="concept-card" style="background:{bg};">
<b>{icon}</b><br><pre style="background:transparent;border:none;font-size:.9rem;">{desc}</pre>
</div>""", unsafe_allow_html=True)

    with tab5:
        st.markdown("### 🧩 적분 퀴즈 6문제")
        st.markdown("---")

        quiz("it1","Q1. ∫₀² 2x dx = ?",
             ["2","4","8","1"],1,
             "F(x)=x². F(2)-F(0) = 4-0 = 4")

        st.markdown("---")
        quiz("it2","Q2. ∫₀¹ x² dx = ?",
             ["1/2","1/3","1","2"],1,
             "F(x)=x³/3. F(1)-F(0) = 1/3-0 = 1/3")

        st.markdown("---")
        quiz("it3","Q3. 리만 합에서 직사각형 수 n이 클수록?",
             ["오차가 커진다","오차가 작아진다","변화 없다","계산이 불가능해진다"],1,
             "직사각형이 많을수록 = 더 잘게 나눌수록 = 정확한 넓이에 가까워져요!")

        st.markdown("---")
        quiz("it4","Q4. 미분과 적분의 관계는?",
             ["같은 연산","서로 역연산","무관한 개념","적분이 미분보다 어렵다"],1,
             "미적분학의 기본 정리: F'(x)=f(x)이면 ∫f(x)dx=F(x). 서로 역연산!")

        st.markdown("---")
        quiz("it5","Q5. x² 를 적분한 결과는?",
             ["2x","x³","x³/3","3x²"],2,
             "∫x²dx = x³/3 + C. 지수에 1 더하고, 새 지수로 나눠요!")

        st.markdown("---")
        quiz("it6","Q6. 자동차가 속도 v(t) = 30 km/h로 2시간 달렸어요. 이동 거리는?",
             ["30km","60km","15km","120km"],1,
             "∫₀² 30 dt = 30×2 = 60km. 속도를 시간으로 적분하면 거리!")

# ══════════════════════════════════════════════════════════════════════════════
#  7장. AI와 미적분
# ══════════════════════════════════════════════════════════════════════════════
def page_ai_calculus():
    st.markdown('<div class="chapter-title">🧠 7장. AI와 미적분 — 딥러닝의 심장</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🤖 <b>이야기: AI가 공부하는 방법</b><br>
ChatGPT 같은 AI는 어떻게 점점 똑똑해질까요?<br>
비밀은 바로 <b>미분</b>이에요!<br>
AI는 처음에 틀린 답을 내요. 그러면 "얼마나 틀렸는지(오차)"를 측정하고,<br>
미분으로 "어떤 방향으로 가면 오차가 줄어드는지" 계산해요.<br>
이 과정을 수백만 번 반복 → AI가 점점 정확해져요! 🎯
</div>""", unsafe_allow_html=True)

    tab1,tab2,tab3=st.tabs(["📖 경사하강법","🎮 직접 체험","🧩 퀴즈"])

    with tab1:
        st.markdown("""
<div class="concept-card">
<h4>✅ 경사하강법(Gradient Descent) — AI 학습의 핵심</h4>
<ol>
<li><b>손실함수(Loss):</b> AI의 오차를 수치로 표현한 함수</li>
<li><b>미분(기울기):</b> "지금 방향에서 어느 쪽으로 가야 오차가 줄어드나?" → 미분으로 계산!</li>
<li><b>업데이트:</b> 기울기 반대 방향으로 조금씩 이동</li>
<li><b>반복:</b> 손실이 0에 가까워질 때까지 계속</li>
</ol>
</div>

<div class="big-formula">새 위치 = 현재 위치 - 학습률 × 기울기(미분값)</div>""",
                    unsafe_allow_html=True)

        col1,col2=st.columns(2)
        col1.markdown("""
<div class="tip-card">
<b>💡 산에서 내려오는 비유</b><br>
안개 낀 산에서 가장 낮은 골짜기를 찾아야 해요.<br>
지금 서있는 곳의 경사(기울기=미분)를 발로 느끼고,<br>
더 낮은 쪽으로 한 발씩 이동해요.<br>
이걸 반복하면 결국 가장 낮은 곳(최솟값)에 도달!
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="concept-card">
<b>📌 적분의 역할</b><br>
AI에서 적분은:<br>
• 확률 계산 (확률밀도함수 적분)<br>
• 평균값 계산<br>
• 데이터의 누적 효과 계산<br>
• 신호 처리 (음성, 이미지)
</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 🎮 경사하강법 시뮬레이션")
        st.write("손실함수 f(x) = x² 에서 기울기(미분)를 이용해 최솟값을 찾아봐요!")

        if "ai_x" not in st.session_state:
            st.session_state.ai_x=4.0
        if "ai_history" not in st.session_state:
            st.session_state.ai_history=[(4.0,16.0)]

        lr=st.slider("학습률 (한 번에 이동 크기)",0.01,0.5,0.1,0.01)

        c1,c2,c3=st.columns(3)
        if c1.button("▶ 한 스텝 학습"):
            grad=2*st.session_state.ai_x  # f'(x)=2x
            st.session_state.ai_x-=lr*grad
            st.session_state.ai_history.append(
                (st.session_state.ai_x, st.session_state.ai_x**2))
        if c2.button("⏩ 10스텝 학습"):
            for _ in range(10):
                grad=2*st.session_state.ai_x
                st.session_state.ai_x-=lr*grad
                st.session_state.ai_history.append(
                    (st.session_state.ai_x, st.session_state.ai_x**2))
        if c3.button("🔄 초기화"):
            st.session_state.ai_x=4.0
            st.session_state.ai_history=[(4.0,16.0)]

        xv=st.session_state.ai_x; yv=xv**2
        col1,col2=st.columns(2)
        col1.metric("현재 x (AI의 파라미터)",f"{xv:.4f}")
        col2.metric("현재 오차 (손실함수 값)",f"{yv:.4f}")

        if abs(xv)<0.01:
            st.success("🎉 AI가 최솟값(x≈0, 오차≈0)에 거의 도달했어요!")
        else:
            st.info(f"💡 미분값 f'({xv:.3f}) = {2*xv:.3f} → 기울기 반대 방향으로 이동 중!")

        fig,ax=plt.subplots(figsize=(9,4.5))
        xc=np.linspace(-5,5,200)
        ax.plot(xc,xc**2,color="#1976D2",lw=2.5,label="손실함수 f(x)=x²")
        hist=st.session_state.ai_history
        hx=[p[0] for p in hist]; hy=[p[1] for p in hist]
        ax.plot(hx,hy,"o-",color="#F44336",markersize=6,lw=1.5,label="AI 학습 경로",alpha=0.7)
        ax.plot(xv,yv,"r*",markersize=18,zorder=6,label=f"현재 위치({xv:.3f},{yv:.3f})")
        ax.set_title(f"경사하강법 — {len(hist)-1}스텝 진행",fontsize=12)
        ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
        ax.set_xlabel("x (파라미터)"); ax.set_ylabel("오차(손실)")
        _show(fig)

    with tab3:
        st.markdown("### 🧩 AI와 미적분 퀴즈")
        st.markdown("---")

        quiz("ai1","Q1. 경사하강법에서 미분(기울기)의 역할은?",
             ["오차를 직접 0으로 만들기","어느 방향으로 가야 오차가 줄어드는지 알려주기",
              "적분을 계산하기","학습률을 결정하기"],1,
             "미분값(기울기)이 오차를 줄이는 방향을 알려줘요!")

        st.markdown("---")
        quiz("ai2","Q2. AI 학습에서 '손실함수(Loss Function)'란?",
             ["AI의 속도","AI의 오차를 수치로 표현한 함수",
              "학습 데이터의 양","모델의 크기"],1,
             "Loss Function은 AI 예측값과 실제 정답의 차이(오차)를 숫자로 표현해요!")

        st.markdown("---")
        quiz("ai3","Q3. f(x) = x²의 최솟값이 있는 x는?",
             ["x=1","x=-1","x=0","x=2"],2,
             "f'(x)=2x=0 → x=0이 최솟값. 미분값=0인 점이 꼭짓점!")

# ══════════════════════════════════════════════════════════════════════════════
#  AI 튜터
# ══════════════════════════════════════════════════════════════════════════════
def page_ai_tutor():
    st.markdown('<div class="chapter-title">🤖 AI 수학 튜터</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="story-box">
🤖 <b>gemma4 AI 튜터에게 무엇이든 물어보세요!</b><br>
수직선, 함수, 기울기, 미분, 적분 — 어떤 것도 OK!<br>
쉬운 말로 친절하게 설명해줘요 😊
</div>""", unsafe_allow_html=True)

    if not OLLAMA_AVAILABLE:
        st.error("⚠️ ollama 없음. `pip install ollama` → `ollama serve`")
        return

    col_m,col_c=st.columns([3,1])
    model=col_m.selectbox("모델",["gemma4:e4b","gemma4:26b"])
    if col_c.button("대화 초기화 🗑️"):
        st.session_state.messages=[]; st.session_state.chat_history=[]
        st.rerun()

    st.markdown("#### 💬 빠른 질문")
    quick=[
        "수직선이 뭐야?","함수를 쉽게 설명해줘",
        "기울기가 뭐야?","미분을 피자로 설명해줘",
        "적분이 뭐야?","미분과 적분 차이가 뭐야?",
        "극한(limit)이 뭔지 알려줘","AI는 미분을 어떻게 쓸까?",
    ]
    cols_q=st.columns(4)
    for i,q in enumerate(quick):
        if cols_q[i%4].button(q,key=f"quick_{i}"):
            st.session_state.pending_question=q

    st.markdown("---")
    if "messages" not in st.session_state: st.session_state.messages=[]
    if "chat_history" not in st.session_state: st.session_state.chat_history=[]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    pending=st.session_state.pop("pending_question",None)
    user_input=st.chat_input("수학에 대해 무엇이든 물어보세요! 🎓")
    question=pending or user_input

    if question:
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("user"): st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("AI가 생각 중... 🤔"):
                answer=ask_ollama(question,model,st.session_state.chat_history)
            st.markdown(f'<div class="ai-bubble">{answer}</div>',unsafe_allow_html=True)
        st.session_state.messages.append({"role":"assistant","content":answer})
        st.session_state.chat_history+=[{"role":"user","content":question},
                                          {"role":"assistant","content":answer}]

    with st.expander("💡 Ollama 설정 안내"):
        st.code("ollama serve\nollama pull gemma4:e4b\nollama pull gemma4:26b",language="bash")

# ══════════════════════════════════════════════════════════════════════════════
#  사이드바 + 라우터
# ══════════════════════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.markdown("""
<div style="text-align:center;padding:10px 0 16px;">
<div style="font-size:2.4rem;">🧮</div>
<div style="font-size:1.15rem;font-weight:900;color:#FFD700;">미적분 탐험대</div>
<div style="font-size:.8rem;color:#aaa;">수포자도 OK!</div>
</div>""", unsafe_allow_html=True)

        selected=st.radio("챕터",list(CHAPTERS.keys()),label_visibility="collapsed")

        st.markdown("---")
        st.markdown("#### 📊 학습 진도")
        chapters_track=["📏 1. 수직선","📊 2. 함숫값","📐 3. 기울기",
                         "🎢 4. 이차함수","⚡ 5. 미분은 기울기다",
                         "🏞️ 6. 적분은 넓이다","🧠 7. AI와 미적분"]
        visited=st.session_state.get("visited_chapters",set())
        done=sum(1 for c in chapters_track if c in visited)
        st.progress(done/7,text=f"{done}/7 완료")
        for ch in chapters_track:
            st.markdown(f"{'✅' if ch in visited else '⬜'} {ch}")

        st.markdown("---")
        st.markdown("#### 🤖 AI 상태")
        if OLLAMA_AVAILABLE: st.success("ollama 설치됨 ✓")
        else: st.error("ollama 미설치")
        return selected

def main():
    selected=sidebar()
    if "visited_chapters" not in st.session_state:
        st.session_state.visited_chapters=set()
    if selected not in ("🏠 홈","🤖 AI 수학 튜터"):
        st.session_state.visited_chapters.add(selected)

    pages={
        "home":page_home,"number_line":page_number_line,
        "function_value":page_function_value,"slope":page_slope,
        "quadratic":page_quadratic,"derivative":page_derivative,
        "integral":page_integral,"ai_calculus":page_ai_calculus,
        "ai_tutor":page_ai_tutor,
    }
    pages[CHAPTERS[selected]]()

if __name__=="__main__":
    main()
