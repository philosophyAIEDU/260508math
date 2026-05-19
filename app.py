"""수포자도 이해하는 수학 탐험대 — 미적분 · 통계학 · 선형대수학 — Streamlit + Ollama(gemma4)"""

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
st.set_page_config(page_title="수학 탐험대 🚀", page_icon="🧮",
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
.kid-box{background:linear-gradient(135deg,#fffde7,#fff8e1);border:3px solid #FFC107;
  border-radius:16px;padding:20px 24px;margin:12px 0;font-size:1.15rem;line-height:2.2;color:#333;}
.ai-connect{background:linear-gradient(135deg,#e3f2fd,#e8eaf6);border:2px solid #3F51B5;
  border-radius:14px;padding:16px;margin:10px 0;color:#1a237e;font-size:1rem;}
</style>
""", unsafe_allow_html=True)

# ── 챕터 정의 ──────────────────────────────────────────────────────────────
CHAPTERS = {
    "🏠 홈": "home",
    # ── 미적분 ──
    "📏 1. 수직선": "number_line",
    "📊 2. 함숫값": "function_value",
    "📐 3. 기울기": "slope",
    "🎢 4. 이차함수": "quadratic",
    "⚡ 5. 미분은 기울기다": "derivative",
    "🏞️ 6. 적분은 넓이다": "integral",
    "🧠 7. AI와 미적분": "ai_calculus",
    # ── 통계학 ──
    "📉 8. 평균과 분산": "stat_descriptive",
    "🔔 9. 확률분포": "stat_distribution",
    "🔗 10. 상관관계와 회귀": "stat_regression",
    # ── 선형대수학 ──
    "🏹 11. 벡터": "linalg_vector",
    "🔲 12. 행렬": "linalg_matrix",
    "🌟 13. 고유값과 고유벡터": "linalg_eigen",
    # ── AI 튜터 ──
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
SYSTEM_PROMPT = """당신은 '수학 탐험대 AI 튜터'입니다. 미적분, 통계학, 선형대수학을 가르칩니다.
규칙:
1. 초등학생도 이해하도록 쉬운 말과 비유를 사용하세요.
2. 예시는 피자, 자전거, 자동차, 온도계 같은 일상 소재로 드세요.
3. 수식은 최대한 단순하게, 말로 풀어서 설명하세요.
4. AI/딥러닝과의 연결을 항상 언급하세요.
5. 항상 응원의 말을 포함하세요.
6. 한국어로 답변하세요."""

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
    🚀 수학 탐험대</h1>
  <p style="font-size:1.3rem;color:#555;">수포자도 OK! 이야기로 배우는 미분·적분·통계·선형대수</p>
</div>""", unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    for col,(icon,title,desc) in zip([c1,c2,c3,c4],[
        ("🎯","단계별 학습","수직선→미적분→통계→선형대수, 순서대로!"),
        ("🎨","풍부한 시각화","모든 개념을 그래프로 눈으로 확인"),
        ("🤖","AI 튜터","gemma4가 모르는 개념 친절하게 설명"),
        ("🧠","AI 연결","각 주제가 AI에 어떻게 쓰이는지 설명"),
    ]):
        col.markdown(f"""<div class="concept-card" style="text-align:center;">
<div style="font-size:2.4rem;">{icon}</div>
<h3 style="color:#1976D2;">{title}</h3>
<p style="color:#555;">{desc}</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ➕ 미적분 (Calculus) — AI 학습의 엔진")
    steps_calc=[("📏","수직선","숫자의 복도"),("📊","함숫값","마법 상자"),
           ("📐","기울기","얼마나 가파를까"),("🎢","이차함수","포물선의 세계"),
           ("⚡","미분","순간 기울기"),("🏞️","적분","곡선 아래 넓이"),("🧠","AI와 미적분","딥러닝의 심장")]
    cols=st.columns(7)
    for col,(icon,name,sub) in zip(cols,steps_calc):
        col.markdown(f"""<div style="text-align:center;background:#e8f4fd;
border-radius:10px;padding:10px 4px;">
<div style="font-size:1.7rem;">{icon}</div>
<div style="font-weight:700;color:#1a237e;font-size:.88rem;">{name}</div>
<div style="font-size:.73rem;color:#777;">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 통계학 (Statistics) — AI가 세상을 이해하는 언어")
    steps_stat=[("📉","평균과 분산","데이터의 중심과 퍼짐"),
                ("🔔","확률분포","정규분포와 확률"),
                ("🔗","상관관계와 회귀","패턴 찾기")]
    cols=st.columns(3)
    for col,(icon,name,sub) in zip(cols,steps_stat):
        col.markdown(f"""<div style="text-align:center;background:#e8f5e9;
border-radius:10px;padding:10px 4px;">
<div style="font-size:1.7rem;">{icon}</div>
<div style="font-weight:700;color:#1b5e20;font-size:.88rem;">{name}</div>
<div style="font-size:.73rem;color:#777;">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔲 선형대수학 (Linear Algebra) — AI의 데이터 언어")
    steps_la=[("🏹","벡터","방향이 있는 화살표"),
              ("🔲","행렬","데이터 변환기"),
              ("🌟","고유값·고유벡터","AI의 핵심 도구")]
    cols=st.columns(3)
    for col,(icon,name,sub) in zip(cols,steps_la):
        col.markdown(f"""<div style="text-align:center;background:#f3e5f5;
border-radius:10px;padding:10px 4px;">
<div style="font-size:1.7rem;">{icon}</div>
<div style="font-weight:700;color:#4a148c;font-size:.88rem;">{name}</div>
<div style="font-size:.73rem;color:#777;">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 사이드바에서 챕터를 선택해 시작하세요! 미적분(1~7장) → 통계학(8~10장) → 선형대수(11~13장) 순서로 읽으면 AI 수학이 완성돼요.")

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
#  통계학 그래프 헬퍼 (초등학생 눈높이)
# ══════════════════════════════════════════════════════════════════════════════

def plot_seesaw_mean(data, title="평균은 시소의 균형점!"):
    """데이터 점들과 평균을 시소 비유로 시각화"""
    mean_v = np.mean(data)
    data_s = sorted(data)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 왼쪽: 점 그래프 + 평균선
    ax = axes[0]
    y_jitter = np.random.RandomState(0).uniform(-0.15, 0.15, len(data))
    ax.scatter(data, y_jitter, s=100, color="#4CAF50", alpha=0.8, zorder=5)
    ax.axvline(mean_v, color="#F44336", lw=4, linestyle="--", label=f"평균 = {mean_v:.1f}", zorder=6)
    ax.set_ylim(-0.6, 0.9); ax.set_yticks([])
    ax.set_xlabel("값", fontsize=12); ax.set_title("점들과 평균", fontsize=13)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.2, axis="x")
    # 화살표로 평균까지 거리 표시
    for v in data[:min(5, len(data))]:
        ax.annotate("", xy=(mean_v, 0.5), xytext=(v, 0.5),
                    arrowprops=dict(arrowstyle="->", color="#FF9800", lw=1.5, alpha=0.7))
    ax.text(mean_v, 0.75, "← 모두 이쪽으로\n균형 잡아요!", ha="center", fontsize=10,
            color="#F44336", fontweight="bold")

    # 오른쪽: 히스토그램 + 통계 표
    ax2 = axes[1]
    ax2.hist(data, bins=min(12, len(data)//2+2), color="#4CAF50", alpha=0.7, edgecolor="white", lw=0.8)
    ax2.axvline(mean_v, color="#F44336", lw=3, linestyle="--", label=f"평균={mean_v:.1f}")
    ax2.axvline(np.median(data), color="#1976D2", lw=2.5, linestyle="-.", label=f"중앙값={np.median(data):.1f}")
    ax2.set_title(title, fontsize=12); ax2.legend(fontsize=10)
    ax2.set_xlabel("값"); ax2.set_ylabel("명수(빈도)"); ax2.grid(True, alpha=0.25)
    std_v = np.std(data)
    ax2.axvspan(mean_v - std_v, mean_v + std_v, alpha=0.12, color="#FF9800",
                label=f"평균±표준편차")
    plt.tight_layout(); return fig

def plot_spread_comparison():
    """두 데이터셋 비교 — 퍼짐의 차이"""
    group_a = [5, 5, 5, 5, 5, 5]
    group_b = [1, 3, 5, 5, 7, 9]
    group_c = [0, 1, 5, 5, 9, 10]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, grp, ttl, col, std_lbl in zip(
        axes,
        [group_a, group_b, group_c],
        ["모두 똑같아요\n(표준편차=0)", "조금 퍼져요\n(표준편차≈2.6)", "많이 퍼져요!\n(표준편차≈3.8)"],
        ["#4CAF50", "#FF9800", "#F44336"],
        ["표준편차 = 0 😊", "표준편차 ≈ 2.6 😐", "표준편차 ≈ 3.8 😮"],
    ):
        y = np.random.RandomState(1).uniform(-0.2, 0.2, len(grp))
        ax.scatter(grp, y, s=200, color=col, alpha=0.85, zorder=5)
        ax.axvline(np.mean(grp), color="#1a237e", lw=3, linestyle="--", label=f"평균={np.mean(grp):.0f}")
        ax.set_xlim(-1, 12); ax.set_ylim(-0.6, 0.8)
        ax.set_yticks([]); ax.set_xlabel("값", fontsize=11)
        ax.set_title(ttl, fontsize=12, color=col, fontweight="bold")
        ax.text(5.5, 0.6, std_lbl, ha="center", fontsize=11, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=col, lw=1.5))
        ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis="x")
    plt.suptitle("평균은 같아도(=5) 퍼짐이 달라요!", fontsize=14, fontweight="bold", color="#1a237e")
    plt.tight_layout(); return fig

def plot_histogram_with_stats(data, title="내 데이터 분석"):
    mean_v = np.mean(data); median_v = np.median(data); std_v = np.std(data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.hist(data, bins=min(15, max(5, len(data)//3)), color="#4CAF50", alpha=0.7, edgecolor="white", lw=0.8)
    ax.axvline(mean_v, color="#F44336", lw=3, linestyle="--", label=f"평균={mean_v:.1f}")
    ax.axvline(median_v, color="#1976D2", lw=2.5, linestyle="-.", label=f"중앙값={median_v:.1f}")
    ax.axvspan(mean_v - std_v, mean_v + std_v, alpha=0.12, color="#FF9800",
                label=f"±표준편차({std_v:.1f})")
    ax.set_title(title, fontsize=12); ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    ax.set_xlabel("값"); ax.set_ylabel("빈도")
    ax2 = axes[1]; ax2.axis("off")
    rows = [
        ["📊 평균 (Mean)", f"{mean_v:.2f}", "가장 대표적인 값"],
        ["📍 중앙값 (Median)", f"{median_v:.2f}", "딱 가운데 값"],
        ["📏 표준편차 (Std)", f"{std_v:.2f}", "얼마나 퍼졌나"],
        ["↔ 범위", f"{np.max(data)-np.min(data):.2f}", "최댓값-최솟값"],
        ["⬇ 최솟값", f"{np.min(data):.2f}", ""],
        ["⬆ 최댓값", f"{np.max(data):.2f}", ""],
    ]
    t = ax2.table(cellText=rows, colLabels=["통계량", "값", "뜻"], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.4, 1.75)
    for (r, c), cell in t.get_celld().items():
        if r == 0: cell.set_facecolor("#1976D2"); cell.set_text_props(color="w", fontweight="bold")
        elif r % 2 == 0: cell.set_facecolor("#e8f5e9")
    ax2.set_title("통계 요약표", fontsize=12)
    plt.tight_layout(); return fig

def plot_normal_distribution(mu=0, sigma=1):
    from scipy.stats import norm
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.linspace(mu - 4.2*sigma, mu + 4.2*sigma, 400)
    y = norm.pdf(x, mu, sigma)
    ax.plot(x, y, color="#1976D2", lw=3)
    fills = [(1,"#4CAF50","68%\n(대부분)"),(2,"#FF9800","95%"),(3,"#F44336","99.7%")]
    for k, col, lbl in reversed(fills):
        xf = np.linspace(mu - k*sigma, mu + k*sigma, 300)
        ax.fill_between(xf, norm.pdf(xf, mu, sigma), alpha=0.22, color=col)
        ax.annotate("", xy=(mu + k*sigma, norm.pdf(mu + k*sigma, mu, sigma)*0.5),
                    xytext=(mu + k*sigma + sigma*0.5, norm.pdf(mu + k*sigma, mu, sigma)*0.5 + 0.02),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
        ax.text(mu + k*sigma + sigma*0.5, norm.pdf(mu + k*sigma, mu, sigma)*0.5 + 0.025,
                lbl, fontsize=10, color=col, fontweight="bold")
    ax.axvline(mu, color="#9C27B0", lw=2.5, linestyle="--")
    ax.text(mu, max(y)*1.05, f"평균\nμ={mu}", ha="center", fontsize=12, color="#9C27B0", fontweight="bold")
    ax.text(mu - sigma, max(y)*0.6, f"σ={sigma}\n(1칸)", ha="center", fontsize=10, color="#555")
    ax.annotate("", xy=(mu, max(y)*0.55), xytext=(mu - sigma, max(y)*0.55),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5))
    ax.set_title(f"🔔 종 모양 분포 (정규분포)  평균={mu}, 표준편차={sigma}", fontsize=13)
    ax.set_xlabel("값 (예: 시험 점수, 키, 몸무게 등)", fontsize=11)
    ax.set_ylabel("이 값이 나올 확률", fontsize=11)
    ax.grid(True, alpha=0.2); ax.set_ylim(0, max(y)*1.18)
    return fig

def plot_scatter_regression(x_data, y_data, x_label="x", y_label="y"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    m, b = np.polyfit(x_data, y_data, 1)
    r = np.corrcoef(x_data, y_data)[0, 1]
    x_line = np.linspace(min(x_data), max(x_data), 100)
    y_pred = m * x_line + b

    ax1 = axes[0]
    ax1.scatter(x_data, y_data, color="#1976D2", s=90, alpha=0.8, zorder=5, label="실제 데이터")
    ax1.plot(x_line, y_pred, color="#F44336", lw=3, label=f"예측선: y={m:.2f}x+{b:.2f}", zorder=4)
    # 오차 화살표 (처음 5개만)
    for xi, yi in zip(list(x_data)[:5], list(y_data)[:5]):
        yi_pred = m * xi + b
        ax1.plot([xi, xi], [yi, yi_pred], color="#FF9800", lw=1.5, alpha=0.7)
    ax1.set_title(f"산점도 & 예측선 (r={r:.2f})", fontsize=12)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.25)
    ax1.set_xlabel(x_label); ax1.set_ylabel(y_label)

    ax2 = axes[1]; ax2.axis("off")
    strength = "아주 강해요 💪" if abs(r) > 0.8 else ("보통이에요" if abs(r) > 0.5 else "약해요")
    direction = "✅ 양의 상관\n(x↑ → y↑)" if r > 0.05 else ("❌ 음의 상관\n(x↑ → y↓)" if r < -0.05 else "무관계")
    rows = [
        ["상관계수 r", f"{r:.3f}"],
        ["관계 강도", strength],
        ["방향", direction],
        ["기울기 a", f"{m:.3f}"],
        ["절편 b", f"{b:.3f}"],
    ]
    t = ax2.table(cellText=rows, colLabels=["항목", "결과"], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1.5, 2.0)
    for (row, c), cell in t.get_celld().items():
        if row == 0: cell.set_facecolor("#F44336"); cell.set_text_props(color="w", fontweight="bold")
        elif row % 2 == 0: cell.set_facecolor("#fff3e0")
    ax2.set_title("회귀 분석 결과", fontsize=12)
    plt.tight_layout()
    return fig, m, b, r

# ══════════════════════════════════════════════════════════════════════════════
#  선형대수학 그래프 헬퍼 (초등학생 눈높이)
# ══════════════════════════════════════════════════════════════════════════════

def plot_treasure_map_vector(vx=3, vy=4):
    """보물찾기 지도 스타일의 벡터 시각화"""
    fig, ax = plt.subplots(figsize=(8, 8))
    # 격자 배경
    for i in range(-1, 7):
        ax.axhline(i, color="#e0e0e0", lw=0.8); ax.axvline(i, color="#e0e0e0", lw=0.8)
    ax.set_facecolor("#f9f9e8")
    # 출발점
    ax.plot(0, 0, "s", color="#1976D2", markersize=18, zorder=6)
    ax.text(0, 0, "🏠", ha="center", va="center", fontsize=14)
    # 도착점(보물)
    ax.plot(vx, vy, "*", color="#FFD700", markersize=28, zorder=6)
    ax.text(vx, vy, "💎", ha="center", va="center", fontsize=13)
    # 벡터 화살표
    ax.annotate("", xy=(vx, vy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#F44336", lw=4))
    # x 이동
    ax.annotate("", xy=(vx, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=2.5,
                                connectionstyle="arc3,rad=0"))
    ax.text(vx/2, -0.5, f"동쪽으로 {vx}칸 →", ha="center", fontsize=13, color="#4CAF50", fontweight="bold")
    # y 이동
    ax.annotate("", xy=(vx, vy), xytext=(vx, 0),
                arrowprops=dict(arrowstyle="->", color="#1976D2", lw=2.5))
    ax.text(vx + 0.3, vy/2, f"북쪽으로\n{vy}칸 ↑", ha="left", fontsize=13, color="#1976D2", fontweight="bold")
    # 크기
    mag = np.sqrt(vx**2 + vy**2)
    ax.text(vx/2 - 0.5, vy/2 + 0.3,
            f"직선거리\n= √({vx}²+{vy}²)\n= {mag:.1f}칸",
            ha="center", fontsize=12, color="#F44336", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#F44336", lw=1.5))
    ax.set_xlim(-0.7, max(vx+1.5, 5)); ax.set_ylim(-1, max(vy+1.5, 5))
    ax.set_xlabel("동쪽 (x)  →", fontsize=12); ax.set_ylabel("북쪽 (y)  ↑", fontsize=12)
    ax.set_title(f"벡터 [{vx}, {vy}] = 동쪽으로 {vx}칸, 북쪽으로 {vy}칸!", fontsize=13)
    return fig

def plot_vectors_2d(vectors, labels, colors, title="벡터"):
    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(-6, 7):
        ax.axhline(i, color="#f0f0f0", lw=0.6); ax.axvline(i, color="#f0f0f0", lw=0.6)
    ax.axhline(0, color="#bbb", lw=1.2); ax.axvline(0, color="#bbb", lw=1.2)
    lim = max(max(abs(v[0]), abs(v[1])) for v in vectors) * 1.45 + 1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    for v, lbl, col in zip(vectors, labels, colors):
        ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=3.2,
                                    mutation_scale=20))
        offset = 0.2
        ax.text(v[0] + offset, v[1] + offset, lbl, fontsize=12, color=col,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.85))
    ax.set_title(title, fontsize=13); ax.grid(False)
    ax.set_xlabel("x (동쪽)"); ax.set_ylabel("y (북쪽)")
    ax.set_aspect("equal")
    return fig

def plot_word_vectors():
    """단어를 벡터로 표현 — AI 임베딩 시각화"""
    words = {"🐱 고양이": [2.1, 1.8], "🐶 강아지": [2.5, 1.3], "🐘 코끼리": [1.2, 3.8],
             "🦁 사자": [1.8, 3.2], "🍎 사과": [-2.0, 1.5], "🍊 오렌지": [-2.4, 1.0]}
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#f8f8ff")
    for i in range(-4, 5):
        ax.axhline(i, color="#eee", lw=0.6); ax.axvline(i, color="#eee", lw=0.6)
    colors_map = {"🐱 고양이":"#F44336","🐶 강아지":"#FF5722","🐘 코끼리":"#9C27B0",
                  "🦁 사자":"#7B1FA2","🍎 사과":"#4CAF50","🍊 오렌지":"#8BC34A"}
    groups = {"동물 🐾":["🐱 고양이","🐶 강아지","🐘 코끼리","🦁 사자"],
              "과일 🍑":["🍎 사과","🍊 오렌지"]}
    for grp_name, grp_words in groups.items():
        pts = np.array([words[w] for w in grp_words])
        cx, cy = pts.mean(axis=0)
        circle = plt.Circle((cx, cy), 0.9, color="#e0e0e0", alpha=0.3, zorder=1)
        ax.add_patch(circle)
        ax.text(cx, cy - 1.2, grp_name, ha="center", fontsize=11, color="#555")
    for word, pos in words.items():
        col = colors_map[word]
        ax.annotate("", xy=(pos[0], pos[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.0, alpha=0.7))
        ax.plot(pos[0], pos[1], "o", color=col, markersize=10, zorder=5)
        ax.text(pos[0]+0.1, pos[1]+0.15, word, fontsize=11, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.9))
    ax.plot(0, 0, "k*", markersize=12, zorder=6); ax.text(0.1, -0.3, "원점(기준)", fontsize=10)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-1, 5)
    ax.set_title("AI는 단어를 화살표(벡터)로 표현해요!\n비슷한 단어 = 비슷한 방향 🎯", fontsize=13)
    ax.set_xlabel("의미 축 1"); ax.set_ylabel("의미 축 2")
    ax.set_aspect("equal"); ax.axhline(0, color="#bbb", lw=1); ax.axvline(0, color="#bbb", lw=1)
    return fig

def plot_matrix_grid_visual(matrix, title="행렬"):
    """행렬을 색깔 격자로 시각화"""
    rows_m, cols_m = matrix.shape
    fig, ax = plt.subplots(figsize=(max(5, cols_m * 1.2), max(4, rows_m * 1.1)))
    vmax = max(abs(matrix.max()), abs(matrix.min()), 1)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(rows_m):
        for j in range(cols_m):
            val = matrix[i, j]
            tc = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=16,
                    fontweight="bold", color=tc)
    ax.set_xticks(range(cols_m)); ax.set_yticks(range(rows_m))
    ax.set_xticklabels([f"열{j+1}" for j in range(cols_m)], fontsize=11)
    ax.set_yticklabels([f"행{i+1}" for i in range(rows_m)], fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout(); return fig

def plot_matrix_transform(matrix, title="행렬 변환"):
    """도형(화살표 등)에 행렬 변환 적용 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # 스마일 모양 포인트들
    theta = np.linspace(0, 2*np.pi, 36)
    circle = np.array([np.cos(theta), np.sin(theta)])
    eye_l = np.array([[-0.4, -0.4, -0.3], [0.5, 0.4, 0.45]])
    eye_r = np.array([[0.3, 0.4, 0.4], [0.5, 0.4, 0.45]])
    mouth_t = np.linspace(-0.5, 0.5, 15)
    mouth = np.array([mouth_t, -0.3 - 0.3 * (mouth_t**2 / 0.25)])
    shapes = [(circle, "#1976D2", 2.5, "얼굴 원"),
              (eye_l, "#333", 3, "왼쪽 눈"),
              (eye_r, "#333", 3, "오른쪽 눈"),
              (mouth, "#F44336", 2.5, "입")]
    for ax, do_transform, ttl in zip(axes, [False, True],
                                      ["변환 전 😊 (원본)", f"변환 후 😲 (행렬 곱)"]):
        ax.set_facecolor("#f9f9f9")
        for pts, col, lw, _ in shapes:
            p = matrix @ pts if do_transform else pts
            ax.plot(p[0], p[1], color=col, lw=lw)
        ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
        lim = 2.5
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(ttl, fontsize=13); ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.set_xlabel("x"); ax.set_ylabel("y")
    axes[1].text(0, -2.2,
                 f"행렬 = [[{matrix[0,0]:.1f},{matrix[0,1]:.1f}],[{matrix[1,0]:.1f},{matrix[1,1]:.1f}]]",
                 ha="center", fontsize=10, color="#555",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"))
    plt.suptitle(title, fontsize=13); plt.tight_layout(); return fig

def plot_eigenvectors(matrix):
    try:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
    except Exception:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors_ev = ["#F44336", "#1976D2"]
    for ax, do_transform, ttl in zip(axes, [False, True],
                                      ["변환 전 (원래 화살표)", "변환 후 (같은 방향! 크기만 변함)"]):
        for i in range(-4, 5):
            ax.axhline(i, color="#f0f0f0", lw=0.5); ax.axvline(i, color="#f0f0f0", lw=0.5)
        ax.axhline(0, color="#bbb", lw=1); ax.axvline(0, color="#bbb", lw=1)
        for idx, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            if np.iscomplex(val): continue
            v = vec.real / (np.linalg.norm(vec.real) + 1e-9) * 2
            p = matrix @ v if do_transform else v
            col = colors_ev[idx % 2]
            ax.annotate("", xy=(p[0], p[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=col, lw=3.5, mutation_scale=22))
            lbl = (f"v{idx+1}: 방향 그대로!\n크기만 {val.real:.1f}배"
                   if do_transform else f"고유벡터 v{idx+1}")
            ax.text(p[0] + 0.15, p[1] + 0.2, lbl, fontsize=10, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.9))
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_title(ttl, fontsize=11); ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.suptitle("🌟 고유벡터 = 행렬이 변환해도 방향이 안 바뀌는 특별한 화살표!", fontsize=13)
    plt.tight_layout(); return fig

# ══════════════════════════════════════════════════════════════════════════════
#  8장. 평균과 분산 (기술통계)
# ══════════════════════════════════════════════════════════════════════════════
def page_stat_descriptive():
    st.markdown('<div class="chapter-title">📉 8장. 평균과 분산 — 데이터의 중심과 퍼짐</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
📊 <b>이야기: 반 아이들의 키</b><br>
우리 반 30명의 키를 측정했어요. 이 많은 숫자를 어떻게 요약할까요?<br>
"평균 키가 165cm" — 단 한 숫자로 전체를 대표할 수 있어요!<br>
하지만 평균만으로는 부족해요. 키가 모두 비슷한지, 아니면 어떤 아이는 매우 크고
어떤 아이는 매우 작은지 모르잖아요.<br>
<b>분산(표준편차)</b>이 바로 "얼마나 퍼져있냐"를 알려줘요! 📏
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 개념", "🔍 단계별 계산", "🎮 직접 해보기", "🧩 퀴즈"])

    with tab1:
        col1, col2 = st.columns(2)
        col1.markdown("""
<div class="concept-card">
<h4>✅ 대표값 — 데이터의 중심</h4>
<ul>
<li><b>평균 (Mean):</b> 모든 값의 합 ÷ 개수<br>
  예) [2, 4, 6] → (2+4+6)/3 = 4</li>
<li><b>중앙값 (Median):</b> 크기 순으로 나열했을 때 가운데 값<br>
  예) [1, 3, 9] → 중앙값 = 3</li>
<li><b>최빈값 (Mode):</b> 가장 자주 나오는 값<br>
  예) [1, 2, 2, 3, 2] → 최빈값 = 2</li>
</ul>
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="concept-card">
<h4>✅ 산포도 — 데이터의 퍼짐</h4>
<ul>
<li><b>분산 (Variance):</b> 각 값과 평균의 차이²의 평균<br>
  → "평균에서 얼마나 멀리 떨어져 있나?"</li>
<li><b>표준편차 (Std Dev):</b> √분산<br>
  → 분산과 같은 단위로 표현</li>
<li><b>범위 (Range):</b> 최댓값 - 최솟값</li>
</ul>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="big-formula">분산 = Σ(값 - 평균)² / n &nbsp;&nbsp;|&nbsp;&nbsp; 표준편차 = √분산</div>""",
                    unsafe_allow_html=True)

        st.markdown("""
<div class="tip-card">
<b>💡 AI에서 통계의 역할</b><br>
• <b>데이터 전처리:</b> 평균=0, 표준편차=1로 정규화 (Normalization) → AI 학습 속도 향상<br>
• <b>이상값 탐지:</b> 평균에서 3σ 이상 떨어진 값은 이상치일 가능성 높음<br>
• <b>손실 측정:</b> MSE(Mean Squared Error) = 분산 개념 그대로!
</div>""", unsafe_allow_html=True)

        np.random.seed(42)
        sample_data = np.concatenate([np.random.normal(170, 8, 80), np.random.normal(155, 5, 20)])
        _show(plot_histogram_with_stats(sample_data, "학생 키 분포 예시 (n=100)"))

    with tab2:
        st.markdown("### 단계별 분산 계산")
        st.markdown("""
<div class="step-card">
<b>📝 예제: [2, 4, 4, 4, 5, 5, 7, 9] 의 분산과 표준편차</b><br>
① 평균 계산: (2+4+4+4+5+5+7+9) ÷ 8 = 40 ÷ 8 = <b>5</b><br>
② 각 값과 평균의 차: -3, -1, -1, -1, 0, 0, 2, 4<br>
③ 차의 제곱: 9, 1, 1, 1, 0, 0, 4, 16<br>
④ 제곱의 평균(분산): (9+1+1+1+0+0+4+16) ÷ 8 = 32 ÷ 8 = <b>4.0</b><br>
⑤ 표준편차: √4.0 = <b>2.0</b>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="warn-card">
<b>⚠️ 평균만 보면 속는다!</b><br>
집합 A = [5, 5, 5, 5] → 평균=5, 표준편차=0 (모두 같음)<br>
집합 B = [1, 3, 7, 9] → 평균=5, 표준편차≈3.2 (많이 퍼짐)<br>
두 집합 모두 평균은 5지만, 데이터의 성격이 완전히 달라요!
</div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("#### 🎮 나만의 데이터로 통계 계산!")
        data_input = st.text_input("숫자를 쉼표로 입력 (예: 70,85,90,60,95,78)", "70,85,90,60,95,78,88,72,65,91")
        try:
            user_data = [float(x.strip()) for x in data_input.split(",") if x.strip()]
            if len(user_data) >= 2:
                _show(plot_histogram_with_stats(user_data, "내 데이터 분석"))
                col1, col2, col3 = st.columns(3)
                col1.metric("평균", f"{np.mean(user_data):.2f}")
                col2.metric("표준편차", f"{np.std(user_data):.2f}")
                col3.metric("중앙값", f"{np.median(user_data):.2f}")
            else:
                st.warning("숫자를 2개 이상 입력해주세요!")
        except Exception:
            st.error("숫자와 쉼표만 입력해주세요!")

        st.markdown("---")
        st.markdown("#### 정규화 (Normalization) 체험")
        st.markdown("AI는 학습 전 데이터를 **평균=0, 표준편차=1**로 변환해요!")
        try:
            if len(user_data) >= 2:
                normalized = (np.array(user_data) - np.mean(user_data)) / np.std(user_data)
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                axes[0].hist(user_data, bins=min(10, len(user_data)), color="#1976D2", alpha=0.7, edgecolor="white")
                axes[0].set_title(f"원본 (평균={np.mean(user_data):.1f})", fontsize=11)
                axes[1].hist(normalized, bins=min(10, len(user_data)), color="#4CAF50", alpha=0.7, edgecolor="white")
                axes[1].set_title(f"정규화 후 (평균≈0, σ≈1)", fontsize=11)
                for ax in axes:
                    ax.grid(True, alpha=0.25); ax.set_xlabel("값"); ax.set_ylabel("빈도")
                plt.tight_layout(); _show(fig)
        except Exception:
            pass

    with tab4:
        st.markdown("### 🧩 기술통계 퀴즈")
        st.markdown("---")
        quiz("stat1", "Q1. [1, 3, 5, 7, 9]의 평균은?",
             ["4", "5", "6", "3"], 1, "합=25, 개수=5 → 25÷5=5!")
        st.markdown("---")
        quiz("stat2", "Q2. 표준편차가 클수록 데이터는?",
             ["평균에 몰려있다", "퍼져있다", "모두 같다", "음수다"], 1,
             "표준편차가 클수록 값들이 평균에서 멀리 떨어져(퍼져) 있어요!")
        st.markdown("---")
        quiz("stat3", "Q3. AI에서 데이터를 평균=0, 표준편차=1로 바꾸는 것은?",
             ["분산", "정규화", "미분", "행렬곱"], 1,
             "정규화(Normalization/Standardization)! AI 학습 속도와 정확도를 높여요!")
        st.markdown("---")
        quiz("stat4", "Q4. MSE(Mean Squared Error)와 관련된 통계 개념은?",
             ["중앙값", "최빈값", "분산", "범위"], 2,
             "MSE = 오차²의 평균 = 분산 개념과 동일! AI 손실함수의 기본이에요.")

# ══════════════════════════════════════════════════════════════════════════════
#  9장. 확률분포
# ══════════════════════════════════════════════════════════════════════════════
def page_stat_distribution():
    st.markdown('<div class="chapter-title">🔔 9장. 확률분포 — AI가 불확실성을 다루는 법</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🎲 <b>이야기: 주사위와 키 분포</b><br>
주사위를 던지면 1~6이 나와요. 각각 나올 확률은 1/6씩 — 이게 <b>확률분포</b>예요!<br>
사람의 키를 측정하면 대부분 평균 근처에 몰리고, 아주 크거나 작은 사람은 드물어요.<br>
이런 패턴을 <b>정규분포(종 모양)</b>라고 해요.<br>
ChatGPT가 다음 단어를 고를 때도 확률분포를 사용해요! 🔔
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 개념", "🔔 정규분포 탐구", "🎮 직접 체험", "🧩 퀴즈"])

    with tab1:
        col1, col2 = st.columns(2)
        col1.markdown("""
<div class="concept-card">
<h4>✅ 확률분포란?</h4>
<ul>
<li><b>이산확률분포:</b> 셀 수 있는 값 (주사위, 동전 등)<br>
  예) 동전 앞면: P=0.5, 뒷면: P=0.5</li>
<li><b>연속확률분포:</b> 연속적인 값 (키, 점수 등)<br>
  예) 키: 165~170cm 사이에 있을 확률</li>
<li><b>핵심 규칙:</b> 모든 확률의 합 = 1<br>
  → 적분하면 1이 돼요!</li>
</ul>
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="concept-card">
<h4>✅ 정규분포 (Normal Distribution)</h4>
<ul>
<li><b>종 모양</b> 그래프 — 자연에서 가장 흔한 분포</li>
<li>μ (뮤) = 평균 → 종의 중심</li>
<li>σ (시그마) = 표준편차 → 종의 폭</li>
<li>μ±1σ 안에 약 <b>68%</b></li>
<li>μ±2σ 안에 약 <b>95%</b></li>
<li>μ±3σ 안에 약 <b>99.7%</b></li>
</ul>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="big-formula">정규분포 PDF: f(x) = (1/σ√2π) × exp(-(x-μ)²/2σ²)</div>""",
                    unsafe_allow_html=True)

        st.markdown("""
<div class="tip-card">
<b>💡 AI와 확률분포</b><br>
• <b>ChatGPT</b>: 다음 토큰(단어)을 확률분포로 선택해요<br>
• <b>이미지 생성 AI (Diffusion Model)</b>: 정규분포 노이즈에서 시작해서 이미지를 만들어요<br>
• <b>Softmax</b>: AI 분류기의 출력을 확률분포로 변환 (합이 1이 되게)
</div>""", unsafe_allow_html=True)

        _show(plot_normal_distribution(0, 1))

    with tab2:
        st.markdown("#### 🔔 정규분포 파라미터 조절")
        c1, c2 = st.columns(2)
        mu_val = c1.slider("평균 μ", -3.0, 3.0, 0.0, 0.5)
        sigma_val = c2.slider("표준편차 σ", 0.3, 3.0, 1.0, 0.1)
        _show(plot_normal_distribution(mu_val, sigma_val))

        col1, col2, col3 = st.columns(3)
        from scipy.stats import norm
        p1 = norm.cdf(mu_val + sigma_val, mu_val, sigma_val) - norm.cdf(mu_val - sigma_val, mu_val, sigma_val)
        p2 = norm.cdf(mu_val + 2*sigma_val, mu_val, sigma_val) - norm.cdf(mu_val - 2*sigma_val, mu_val, sigma_val)
        col1.metric(f"μ±1σ 범위 내 확률", f"{p1*100:.1f}%")
        col2.metric(f"μ±2σ 범위 내 확률", f"{p2*100:.1f}%")
        col3.metric("μ 이하 확률 (50%)", f"{norm.cdf(mu_val, mu_val, sigma_val)*100:.1f}%")

        st.markdown("""
<div class="warn-card">
<b>⚠️ 왜 정규분포가 중요한가?</b><br>
중심극한정리: 어떤 분포든 충분히 많이 더하면 정규분포에 가까워져요!<br>
그래서 현실 데이터의 오차(noise)는 대부분 정규분포를 따라요.
</div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("#### 🎲 주사위 시뮬레이션 — 중심극한정리")
        n_dice = st.slider("주사위 개수 (더 많을수록 정규분포에 가까워져요!)", 1, 20, 1)
        n_trials = 3000
        np.random.seed(0)
        results = np.sum(np.random.randint(1, 7, (n_trials, n_dice)), axis=1)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(results, bins=min(30, n_dice*5+5), color="#9C27B0", alpha=0.7, edgecolor="white",
                density=True, label=f"주사위 {n_dice}개 합 ({n_trials}번)")
        if n_dice >= 3:
            mu_est, sigma_est = np.mean(results), np.std(results)
            x_norm = np.linspace(results.min()-1, results.max()+1, 200)
            from scipy.stats import norm
            ax.plot(x_norm, norm.pdf(x_norm, mu_est, sigma_est), color="#F44336", lw=2.5, label="정규분포 근사")
        ax.set_title(f"주사위 {n_dice}개의 합 분포 — {'정규분포에 가까워요!' if n_dice>=5 else '아직 균일에 가까워요'}", fontsize=11)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
        ax.set_xlabel("합계"); ax.set_ylabel("확률밀도")
        _show(fig)
        st.info(f"주사위가 {n_dice}개일 때: 평균≈{np.mean(results):.1f}, 표준편차≈{np.std(results):.1f}")

    with tab4:
        st.markdown("### 🧩 확률분포 퀴즈")
        st.markdown("---")
        quiz("dist1", "Q1. 정규분포에서 μ±2σ 안에 데이터가 약 몇 % 있나요?",
             ["68%", "95%", "99.7%", "50%"], 1, "68-95-99.7 규칙! μ±2σ = 약 95%!")
        st.markdown("---")
        quiz("dist2", "Q2. ChatGPT가 다음 단어를 선택할 때 사용하는 것은?",
             ["평균값", "확률분포", "분산", "표준편차"], 1,
             "ChatGPT는 각 토큰(단어)의 확률분포(Softmax 출력)에서 다음 단어를 선택해요!")
        st.markdown("---")
        quiz("dist3", "Q3. 정규분포를 결정하는 두 파라미터는?",
             ["최솟값과 최댓값", "평균(μ)과 표준편차(σ)", "중앙값과 최빈값", "분산과 범위"], 1,
             "정규분포 N(μ, σ²)는 평균 μ와 표준편차 σ로 완전히 결정돼요!")
        st.markdown("---")
        quiz("dist4", "Q4. 확률분포에서 전체 면적(적분값)은?",
             ["0", "0.5", "1", "무한대"], 2,
             "확률의 총합은 항상 1! 연속분포에서는 넓이(적분) = 1이에요.")

# ══════════════════════════════════════════════════════════════════════════════
#  10장. 상관관계와 회귀
# ══════════════════════════════════════════════════════════════════════════════
def page_stat_regression():
    st.markdown('<div class="chapter-title">🔗 10장. 상관관계와 회귀 — AI 예측의 기초</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🏠 <b>이야기: 집 크기와 가격</b><br>
집이 클수록 비싸다 — 이런 패턴을 <b>상관관계</b>라고 해요.<br>
그리고 "집 크기로 가격을 예측하는 수식"을 찾는 것이 <b>회귀(Regression)</b>예요!<br>
AI 예측 모델의 가장 기본 원리: 데이터에서 패턴을 찾아 <b>직선(또는 곡선)으로 표현</b>해요.<br>
이 직선을 찾는 데 미분(경사하강법)이 사용돼요! 🏡
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 개념", "🔍 상관계수 탐구", "🎮 직접 해보기", "🧩 퀴즈"])

    with tab1:
        col1, col2 = st.columns(2)
        col1.markdown("""
<div class="concept-card">
<h4>✅ 상관계수 r</h4>
<ul>
<li><b>r = +1:</b> 완전한 양의 상관 (x↑ → y↑)</li>
<li><b>r = 0:</b> 상관없음 (x와 y 무관계)</li>
<li><b>r = -1:</b> 완전한 음의 상관 (x↑ → y↓)</li>
<li>|r| > 0.7: 강한 상관관계</li>
<li>|r| < 0.3: 약한 상관관계</li>
</ul>
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="concept-card">
<h4>✅ 선형회귀 (Linear Regression)</h4>
<ul>
<li><b>목표:</b> y = ax + b 직선 찾기</li>
<li><b>방법:</b> 예측값과 실제값의 차이(MSE)를 최소화</li>
<li><b>AI 연결:</b> 가장 단순한 AI 예측 모델</li>
<li>딥러닝도 이 아이디어의 확장이에요!</li>
</ul>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="big-formula">MSE = (1/n) Σ(y예측 - y실제)² → 이것을 최소화하는 a, b를 찾자!</div>""",
                    unsafe_allow_html=True)

        np.random.seed(7)
        x_demo = np.random.uniform(20, 80, 40)
        y_demo = 2.5 * x_demo + 10 + np.random.normal(0, 15, 40)
        fig_demo, _, _, _ = plot_scatter_regression(x_demo, y_demo, "집 크기 (평)", "가격 (천만원)")
        _show(fig_demo)

        st.markdown("""
<div class="tip-card">
<b>💡 AI 모델링의 흐름</b><br>
① 데이터 수집 → ② 상관관계 확인 → ③ 회귀 모델 선택<br>
④ 경사하강법으로 최적 파라미터(a, b) 학습 → ⑤ 예측!<br>
딥러닝은 이 과정을 수천 개의 변수에 대해 반복하는 것이에요.
</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 🔍 상관계수에 따른 산점도 모양")
        target_r = st.select_slider("목표 상관계수 r", [-0.95, -0.7, -0.4, 0.0, 0.4, 0.7, 0.95], value=0.7)
        np.random.seed(42)
        n_pts = 50
        cov_mat = [[1, target_r], [target_r, 1]]
        xy = np.random.multivariate_normal([0, 0], cov_mat, n_pts)
        fig_r, ax_r = plt.subplots(figsize=(7, 5))
        ax_r.scatter(xy[:, 0], xy[:, 1], color="#1976D2", s=60, alpha=0.8)
        if abs(target_r) > 0.1:
            m_r, b_r = np.polyfit(xy[:, 0], xy[:, 1], 1)
            x_line = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 100)
            ax_r.plot(x_line, m_r * x_line + b_r, color="#F44336", lw=2.5)
        actual_r = np.corrcoef(xy[:, 0], xy[:, 1])[0, 1]
        strength = "강한" if abs(actual_r) > 0.7 else ("보통" if abs(actual_r) > 0.4 else "약한")
        direction = "양의" if actual_r > 0.05 else ("음의" if actual_r < -0.05 else "")
        ax_r.set_title(f"r = {actual_r:.3f} ({strength} {direction} 상관관계)", fontsize=12)
        ax_r.grid(True, alpha=0.25); ax_r.set_xlabel("x"); ax_r.set_ylabel("y")
        _show(fig_r)

    with tab3:
        st.markdown("#### 🎮 나만의 데이터로 회귀 분석!")
        preset = st.selectbox("예시 데이터 선택", ["공부시간 vs 점수", "온도 vs 아이스크림 판매량", "키 vs 몸무게", "직접 입력"])
        if preset == "공부시간 vs 점수":
            x_d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
            y_d = np.array([45, 52, 58, 65, 70, 78, 82, 88, 91, 96], dtype=float)
            xl, yl = "공부시간 (시간)", "시험 점수"
        elif preset == "온도 vs 아이스크림 판매량":
            x_d = np.array([10, 15, 18, 22, 25, 28, 30, 33, 35, 38], dtype=float)
            y_d = np.array([20, 35, 55, 90, 130, 175, 210, 260, 300, 350], dtype=float)
            xl, yl = "기온 (°C)", "아이스크림 판매량 (개)"
        elif preset == "키 vs 몸무게":
            np.random.seed(5)
            x_d = np.random.uniform(155, 185, 20)
            y_d = (x_d - 100) + np.random.normal(0, 5, 20)
            xl, yl = "키 (cm)", "몸무게 (kg)"
        else:
            x_str = st.text_input("x 값들 (쉼표 구분)", "1,2,3,4,5,6,7")
            y_str = st.text_input("y 값들 (쉼표 구분)", "2,4,5,4,5,7,8")
            try:
                x_d = np.array([float(v.strip()) for v in x_str.split(",")])
                y_d = np.array([float(v.strip()) for v in y_str.split(",")])
                xl, yl = "x", "y"
            except Exception:
                st.error("숫자를 올바르게 입력해주세요!"); return

        if len(x_d) >= 2 and len(x_d) == len(y_d):
            fig_reg, m_reg, b_reg, r_reg = plot_scatter_regression(x_d, y_d, xl, yl)
            _show(fig_reg)
            col1, col2, col3 = st.columns(3)
            col1.metric("회귀 기울기 a", f"{m_reg:.3f}")
            col2.metric("y절편 b", f"{b_reg:.3f}")
            col3.metric("상관계수 r", f"{r_reg:.3f}")
            pred_x = st.number_input(f"{xl} 값으로 {yl} 예측", value=float(np.mean(x_d)))
            pred_y = m_reg * pred_x + b_reg
            st.success(f"예측: {xl}={pred_x:.1f} → {yl} ≈ **{pred_y:.2f}**")

    with tab4:
        st.markdown("### 🧩 상관관계와 회귀 퀴즈")
        st.markdown("---")
        quiz("reg1", "Q1. 상관계수 r = -0.9의 의미는?",
             ["강한 양의 상관", "약한 상관", "강한 음의 상관", "상관없음"], 2,
             "r이 -1에 가까울수록 강한 음의 상관관계! x가 커지면 y가 작아져요.")
        st.markdown("---")
        quiz("reg2", "Q2. 선형회귀에서 MSE를 최소화하는 방법은?",
             ["적분", "경사하강법 (미분 활용)", "표준편차 계산", "평균 계산"], 1,
             "MSE를 파라미터에 대해 미분 → 기울기 반대 방향으로 이동 = 경사하강법!")
        st.markdown("---")
        quiz("reg3", "Q3. y = 2x + 3 회귀선에서 x=5일 때 y 예측값은?",
             ["10", "13", "15", "8"], 1, "y = 2×5 + 3 = 10 + 3 = 13!")
        st.markdown("---")
        quiz("reg4", "Q4. 딥러닝(신경망)은 선형회귀와 어떤 관계인가요?",
             ["완전히 다른 원리", "선형회귀를 수천 개 레이어에 확장한 것", "통계와 무관", "적분만 사용"], 1,
             "딥러닝은 선형변환 + 비선형 활성화의 조합! 선형회귀가 기초에요.")

# ══════════════════════════════════════════════════════════════════════════════
#  11장. 벡터
# ══════════════════════════════════════════════════════════════════════════════
def page_linalg_vector():
    st.markdown('<div class="chapter-title">🏹 11장. 벡터 — 방향이 있는 화살표</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🗺️ <b>이야기: 보물찾기 지도</b><br>
"동쪽으로 3칸, 북쪽으로 4칸 이동하면 보물이 있어요!" — 이게 바로 <b>벡터</b>예요!<br>
벡터는 <b>크기</b>와 <b>방향</b>을 동시에 가진 화살표예요.<br>
AI에서 단어, 이미지, 사용자 취향 — 모든 것이 벡터로 표현돼요.<br>
ChatGPT의 "임베딩"도 단어를 1000차원 벡터로 변환하는 거예요! 🗺️
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 개념", "🔍 벡터 연산", "🎮 직접 해보기", "🧩 퀴즈"])

    with tab1:
        col1, col2 = st.columns(2)
        col1.markdown("""
<div class="concept-card">
<h4>✅ 벡터란?</h4>
<ul>
<li><b>스칼라 (Scalar):</b> 크기만 있음 (온도, 나이, 점수)</li>
<li><b>벡터 (Vector):</b> 크기 + 방향 (속도, 힘, 위치 이동)</li>
<li><b>표현:</b> v = [3, 4] → x방향으로 3, y방향으로 4</li>
<li><b>크기(norm):</b> |v| = √(3² + 4²) = √25 = 5</li>
</ul>
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="concept-card">
<h4>✅ 벡터 연산</h4>
<ul>
<li><b>덧셈:</b> [1,2] + [3,1] = [4,3]</li>
<li><b>스칼라 곱:</b> 2 × [3,4] = [6,8] (크기 2배, 방향 유지)</li>
<li><b>내적 (Dot Product):</b> [a,b]·[c,d] = ac + bd<br>
  → 두 벡터의 유사도 측정!</li>
<li><b>코사인 유사도:</b> cos θ = (a·b) / (|a||b|)</li>
</ul>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="big-formula">내적: a · b = |a||b|cos(θ) &nbsp;&nbsp;→ θ=0°이면 같은 방향, θ=90°이면 수직</div>""",
                    unsafe_allow_html=True)

        _show(plot_vectors_2d([[3, 4], [1, 2], [4, 6]], ["a=[3,4]", "b=[1,2]", "a+b=[4,6]"],
                              ["#F44336", "#1976D2", "#4CAF50"], "벡터 덧셈 시각화"))

        st.markdown("""
<div class="tip-card">
<b>💡 AI에서 벡터의 역할</b><br>
• <b>Word2Vec / 임베딩:</b> 단어 → 벡터 (비슷한 단어 = 비슷한 벡터 방향)<br>
• <b>추천 시스템:</b> 사용자 취향 = 벡터, 내적으로 유사도 계산<br>
• <b>검색:</b> 질문 벡터와 문서 벡터의 코사인 유사도로 관련 문서 찾기<br>
• <b>신경망:</b> 각 레이어의 데이터가 벡터로 이동
</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 🔍 벡터 덧셈과 내적 단계별")
        st.markdown("""
<div class="step-card">
<b>📝 벡터 덧셈 예제</b><br>
a = [2, 3], b = [4, -1]<br>
a + b = [2+4, 3+(-1)] = [6, 2]<br>
→ 각 성분끼리 더해요!
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div class="step-card">
<b>📝 내적 예제 — 유사도 측정</b><br>
a = [1, 0] (동쪽만), b = [0, 1] (북쪽만)<br>
a · b = 1×0 + 0×1 = 0 → 수직! 서로 전혀 다른 방향<br><br>
a = [1, 0], c = [0.8, 0.6]<br>
a · c = 1×0.8 + 0×0.6 = 0.8 → 0에 가까울수록 다르고, 1에 가까울수록 비슷!
</div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("#### 🎮 나만의 벡터 탐험!")
        c1, c2 = st.columns(2)
        ax_v = c1.slider("벡터 a의 x성분", -5, 5, 3)
        ay_v = c1.slider("벡터 a의 y성분", -5, 5, 2)
        bx_v = c2.slider("벡터 b의 x성분", -5, 5, -1)
        by_v = c2.slider("벡터 b의 y성분", -5, 5, 3)

        sum_v = [ax_v + bx_v, ay_v + by_v]
        dot_v = ax_v * bx_v + ay_v * by_v
        mag_a = np.sqrt(ax_v**2 + ay_v**2)
        mag_b = np.sqrt(bx_v**2 + by_v**2)
        cos_sim = dot_v / (mag_a * mag_b) if mag_a > 0 and mag_b > 0 else 0

        vecs = [[ax_v, ay_v], [bx_v, by_v], sum_v]
        labels = [f"a=[{ax_v},{ay_v}]", f"b=[{bx_v},{by_v}]", f"a+b={sum_v}"]
        colors = ["#F44336", "#1976D2", "#4CAF50"]
        _show(plot_vectors_2d(vecs, labels, colors))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("|a| (크기)", f"{mag_a:.2f}")
        col2.metric("|b| (크기)", f"{mag_b:.2f}")
        col3.metric("a · b (내적)", f"{dot_v:.2f}")
        col4.metric("코사인 유사도", f"{cos_sim:.3f}")

        angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1))) if mag_a > 0 and mag_b > 0 else 90
        msg = f"두 벡터 사이 각도: {angle:.1f}° "
        if angle < 30: msg += "— 거의 같은 방향이에요! (AI에서 의미가 비슷)"
        elif angle < 90: msg += "— 비슷한 방향이에요"
        elif angle < 120: msg += "— 어느 정도 다른 방향이에요"
        else: msg += "— 반대 방향에 가까워요 (AI에서 의미가 반대)"
        st.info(msg)

    with tab4:
        st.markdown("### 🧩 벡터 퀴즈")
        st.markdown("---")
        quiz("vec1", "Q1. 벡터 [3, 4]의 크기(norm)는?",
             ["7", "5", "12", "25"], 1, "√(3²+4²) = √(9+16) = √25 = 5!")
        st.markdown("---")
        quiz("vec2", "Q2. 내적(dot product)이 0이면 두 벡터는?",
             ["같은 방향", "반대 방향", "수직(직교)", "크기가 같음"], 2,
             "a·b = |a||b|cos(90°) = 0 → 수직! 서로 전혀 다른 방향이에요.")
        st.markdown("---")
        quiz("vec3", "Q3. AI에서 단어를 벡터로 변환하는 기술은?",
             ["미분", "임베딩 (Embedding)", "적분", "분산"], 1,
             "Word2Vec, BERT 등은 단어를 고차원 벡터로 변환해요. 비슷한 단어 = 비슷한 벡터!")
        st.markdown("---")
        quiz("vec4", "Q4. [1,2] + [3,4] = ?",
             ["[2,6]", "[4,6]", "[3,8]", "[4,8]"], 1, "성분별로 더하면 [1+3, 2+4] = [4, 6]!")

# ══════════════════════════════════════════════════════════════════════════════
#  12장. 행렬
# ══════════════════════════════════════════════════════════════════════════════
def page_linalg_matrix():
    st.markdown('<div class="chapter-title">🔲 12장. 행렬 — 데이터 변환의 마법</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
📸 <b>이야기: 사진을 돌리거나 늘리는 마법</b><br>
사진 편집 앱에서 이미지를 회전하고, 크기를 바꾸고, 반전시킬 수 있어요.<br>
이런 모든 변환이 <b>행렬 곱셈</b>으로 이루어져요!<br>
신경망의 각 레이어도 입력 벡터에 <b>가중치 행렬을 곱하는</b> 연산이에요.<br>
행렬을 이해하면 AI의 내부 동작 원리가 보여요! 🔲
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 개념", "🔍 행렬 연산", "🎮 직접 변환", "🧩 퀴즈"])

    with tab1:
        col1, col2 = st.columns(2)
        col1.markdown("""
<div class="concept-card">
<h4>✅ 행렬이란?</h4>
<ul>
<li><b>행렬 (Matrix):</b> 숫자를 직사각형으로 배열한 것</li>
<li>m행 × n열 행렬 = m×n 크기</li>
<li>AI에서 이미지 = 픽셀값 행렬</li>
<li>AI 가중치 = 수백만 개의 행렬</li>
</ul>
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="concept-card">
<h4>✅ 행렬 연산</h4>
<ul>
<li><b>덧셈:</b> 같은 위치 원소끼리 더함</li>
<li><b>스칼라 곱:</b> 모든 원소에 스칼라 곱</li>
<li><b>행렬 곱 (가장 중요!):</b> A(m×k) × B(k×n) = C(m×n)</li>
<li><b>전치 (Transpose):</b> 행과 열을 바꿈 A → Aᵀ</li>
</ul>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="big-formula">행렬 곱: C[i,j] = Σₖ A[i,k] × B[k,j] &nbsp;&nbsp;→ 신경망의 핵심 연산!</div>""",
                    unsafe_allow_html=True)

        st.markdown("""
<div class="step-card">
<b>📝 2×2 행렬 곱 예제</b><br>
A = [[1,2],[3,4]], B = [[5,6],[7,8]]<br>
C[0,0] = 1×5 + 2×7 = 5+14 = 19<br>
C[0,1] = 1×6 + 2×8 = 6+16 = 22<br>
C[1,0] = 3×5 + 4×7 = 15+28 = 43<br>
C[1,1] = 3×6 + 4×8 = 18+32 = 50<br>
→ C = [[19,22],[43,50]]
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="tip-card">
<b>💡 신경망에서의 행렬</b><br>
• 입력 데이터 x = [x₁, x₂, ..., xₙ] (벡터)<br>
• 가중치 W = 행렬 (수백만 개의 숫자)<br>
• 레이어 출력 = W × x + b (행렬 곱 + 편향 벡터)<br>
• 이 과정을 수십~수백 레이어 반복 = 딥러닝!
</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 🔍 2×2 행렬 연산 계산기")
        c1, c2 = st.columns(2)
        c1.markdown("**행렬 A**")
        a11 = c1.number_input("A[0,0]", value=1.0, step=0.5, key="a11")
        a12 = c1.number_input("A[0,1]", value=2.0, step=0.5, key="a12")
        a21 = c1.number_input("A[1,0]", value=0.0, step=0.5, key="a21")
        a22 = c1.number_input("A[1,1]", value=1.0, step=0.5, key="a22")
        c2.markdown("**행렬 B**")
        b11 = c2.number_input("B[0,0]", value=2.0, step=0.5, key="b11")
        b12 = c2.number_input("B[0,1]", value=0.0, step=0.5, key="b12")
        b21 = c2.number_input("B[1,0]", value=0.0, step=0.5, key="b21")
        b22 = c2.number_input("B[1,1]", value=2.0, step=0.5, key="b22")

        A = np.array([[a11, a12], [a21, a22]])
        B = np.array([[b11, b12], [b21, b22]])
        C = A @ B
        det_A = np.linalg.det(A)

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"""<div class="concept-card" style="text-align:center;">
<b>A × B =</b><br>
[[{C[0,0]:.2f}, {C[0,1]:.2f}],<br>
 [{C[1,0]:.2f}, {C[1,1]:.2f}]]
</div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div class="concept-card" style="text-align:center;">
<b>det(A) =</b><br>
{det_A:.3f}<br>
{'(역행렬 존재 ✓)' if abs(det_A) > 0.001 else '(역행렬 없음 ✗)'}
</div>""", unsafe_allow_html=True)
        AT = A.T
        col3.markdown(f"""<div class="concept-card" style="text-align:center;">
<b>Aᵀ (전치) =</b><br>
[[{AT[0,0]:.1f}, {AT[0,1]:.1f}],<br>
 [{AT[1,0]:.1f}, {AT[1,1]:.1f}]]
</div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("#### 🎮 행렬 변환 시각화")
        transform_type = st.selectbox("변환 종류 선택", [
            "항등 변환 (변화 없음)",
            "2배 확대",
            "x축 반전",
            "90° 회전",
            "기울이기 (Shear)",
            "직접 입력",
        ])
        if transform_type == "항등 변환 (변화 없음)":
            M = np.array([[1.0, 0.0], [0.0, 1.0]])
        elif transform_type == "2배 확대":
            M = np.array([[2.0, 0.0], [0.0, 2.0]])
        elif transform_type == "x축 반전":
            M = np.array([[1.0, 0.0], [0.0, -1.0]])
        elif transform_type == "90° 회전":
            M = np.array([[0.0, -1.0], [1.0, 0.0]])
        elif transform_type == "기울이기 (Shear)":
            M = np.array([[1.0, 1.0], [0.0, 1.0]])
        else:
            cc1, cc2 = st.columns(2)
            m11 = cc1.number_input("M[0,0]", value=1.0, step=0.5, key="m11")
            m12 = cc1.number_input("M[0,1]", value=0.0, step=0.5, key="m12")
            m21 = cc2.number_input("M[1,0]", value=0.0, step=0.5, key="m21")
            m22 = cc2.number_input("M[1,1]", value=1.0, step=0.5, key="m22")
            M = np.array([[m11, m12], [m21, m22]])

        _show(plot_matrix_transform(M, f"행렬 변환: {M.tolist()}"))
        det_M = np.linalg.det(M)
        st.info(f"행렬식 det(M) = {det_M:.3f} → 변환 후 넓이는 원본의 {abs(det_M):.2f}배 {'(반전 있음)' if det_M < 0 else ''}")

    with tab4:
        st.markdown("### 🧩 행렬 퀴즈")
        st.markdown("---")
        quiz("mat1", "Q1. 행렬 곱 A(2×3) × B(3×4) 의 결과 크기는?",
             ["2×4", "3×3", "2×3", "4×2"], 0, "A(m×k) × B(k×n) = C(m×n) → 2×4!")
        st.markdown("---")
        quiz("mat2", "Q2. 신경망의 각 레이어에서 핵심 연산은?",
             ["미분", "적분", "행렬 곱", "벡터 덧셈만"], 2,
             "레이어 출력 = 가중치 행렬 W × 입력 벡터 x + 편향 b! 행렬 곱이 핵심이에요.")
        st.markdown("---")
        quiz("mat3", "Q3. 단위 행렬(Identity Matrix) I와 곱하면?",
             ["0이 됨", "원래 행렬이 됨", "전치됨", "역행렬이 됨"], 1,
             "A × I = I × A = A! 단위 행렬은 1과 같은 역할을 해요.")
        st.markdown("---")
        quiz("mat4", "Q4. 이미지를 90° 회전시키는 변환 행렬은?",
             ["[[1,0],[0,1]]", "[[0,-1],[1,0]]", "[[2,0],[0,2]]", "[[1,1],[0,1]]"], 1,
             "회전 행렬 [[cos90°,-sin90°],[sin90°,cos90°]] = [[0,-1],[1,0]]!")

# ══════════════════════════════════════════════════════════════════════════════
#  13장. 고유값과 고유벡터
# ══════════════════════════════════════════════════════════════════════════════
def page_linalg_eigen():
    st.markdown('<div class="chapter-title">🌟 13장. 고유값과 고유벡터 — AI의 숨겨진 보석</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="story-box">
🔭 <b>이야기: 변환해도 방향이 안 바뀌는 마법의 벡터</b><br>
어떤 행렬로 변환해도 <b>방향은 그대로, 크기만 바뀌는</b> 특별한 벡터가 있어요.<br>
이것이 <b>고유벡터(Eigenvector)</b>이고, 크기가 얼마나 바뀌는지가 <b>고유값(Eigenvalue)</b>이에요.<br>
PCA(주성분 분석), 추천 시스템, 구글의 PageRank — 모두 고유값 분해를 사용해요!<br>
데이터의 가장 중요한 방향을 찾는 AI의 핵심 도구예요 🌟
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 개념", "🔍 고유값 탐구", "🎮 PCA 체험", "🧩 퀴즈"])

    with tab1:
        col1, col2 = st.columns(2)
        col1.markdown("""
<div class="concept-card">
<h4>✅ 고유값/고유벡터 정의</h4>
<ul>
<li><b>정의:</b> Av = λv<br>
  행렬 A를 곱해도 방향 불변!</li>
<li><b>v:</b> 고유벡터 (방향 유지)</li>
<li><b>λ (람다):</b> 고유값 (크기 변화 비율)</li>
<li>λ > 1: 늘어남 / λ < 1: 줄어듦<br>
  λ < 0: 반전!</li>
</ul>
</div>""", unsafe_allow_html=True)
        col2.markdown("""
<div class="concept-card">
<h4>✅ 왜 중요한가?</h4>
<ul>
<li><b>PCA:</b> 데이터의 주요 방향 찾기 → 차원 축소</li>
<li><b>SVD:</b> 추천 시스템, 이미지 압축</li>
<li><b>PageRank:</b> 구글이 웹페이지 중요도 계산</li>
<li><b>진동 분석:</b> 공명 주파수 = 고유값</li>
</ul>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="big-formula">Av = λv &nbsp;&nbsp;→ (A - λI)v = 0 &nbsp;&nbsp;→ det(A - λI) = 0 으로 λ 계산</div>""",
                    unsafe_allow_html=True)

        st.markdown("""
<div class="tip-card">
<b>💡 PCA (주성분 분석) — AI의 필수 전처리</b><br>
• 100개의 특성(feature)을 가진 데이터 → 가장 중요한 2~3개 방향으로 압축<br>
• 각 방향 = 데이터 분산이 최대인 방향 = 공분산 행렬의 <b>고유벡터</b><br>
• 각 방향의 중요도 = <b>고유값</b> (클수록 중요)<br>
• 얼굴 인식, 자연어 처리, 이상 탐지에 사용!
</div>""", unsafe_allow_html=True)

        M_ex = np.array([[3.0, 1.0], [1.0, 3.0]])
        fig_ev = plot_eigenvectors(M_ex)
        if fig_ev: _show(fig_ev)

    with tab2:
        st.markdown("#### 🔍 행렬의 고유값 직접 계산")
        c1, c2 = st.columns(2)
        e11 = c1.number_input("M[0,0]", value=3.0, step=0.5, key="e11")
        e12 = c1.number_input("M[0,1]", value=1.0, step=0.5, key="e12")
        e21 = c2.number_input("M[1,0]", value=1.0, step=0.5, key="e21")
        e22 = c2.number_input("M[1,1]", value=3.0, step=0.5, key="e22")

        M_user = np.array([[e11, e12], [e21, e22]])
        eigenvalues, eigenvectors = np.linalg.eig(M_user)

        st.markdown(f"""
<div class="concept-card">
<b>행렬 M = [[{e11}, {e12}], [{e21}, {e22}]]의 고유값 분해</b><br><br>
""" + "".join([
            f"고유값 λ{i+1} = <span class='highlight'>{val.real:.3f}</span> &nbsp;|&nbsp; "
            f"고유벡터 v{i+1} = [{eigenvectors[0,i].real:.3f}, {eigenvectors[1,i].real:.3f}]<br>"
            for i, val in enumerate(eigenvalues)
        ]) + """<br>
검증: M × v1 = λ1 × v1 ?
</div>""", unsafe_allow_html=True)

        fig_ev2 = plot_eigenvectors(M_user)
        if fig_ev2: _show(fig_ev2)

        v1 = eigenvectors[:, 0].real
        Mv1 = M_user @ v1
        lambda1 = eigenvalues[0].real
        st.info(f"검증: M × v1 = [{Mv1[0]:.3f}, {Mv1[1]:.3f}], λ1 × v1 = [{lambda1*v1[0]:.3f}, {lambda1*v1[1]:.3f}] ✓")

    with tab3:
        st.markdown("#### 🎮 PCA (주성분 분석) 체험")
        st.write("2차원 데이터에서 가장 중요한 방향(주성분)을 고유벡터로 찾아봐요!")

        data_type = st.selectbox("데이터 종류", ["길쭉한 분포", "둥근 분포", "사선 분포"])
        np.random.seed(42)
        if data_type == "길쭉한 분포":
            cov = [[4, 0], [0, 0.5]]
        elif data_type == "둥근 분포":
            cov = [[2, 0], [0, 2]]
        else:
            cov = [[2, 1.5], [1.5, 2]]

        data_pca = np.random.multivariate_normal([0, 0], cov, 100)
        cov_matrix = np.cov(data_pca.T)
        eigenvalues_pca, eigenvectors_pca = np.linalg.eig(cov_matrix)
        order = np.argsort(eigenvalues_pca)[::-1]
        eigenvalues_pca = eigenvalues_pca[order]
        eigenvectors_pca = eigenvectors_pca[:, order]

        fig_pca, axes_pca = plt.subplots(1, 2, figsize=(12, 5))
        ax1 = axes_pca[0]
        ax1.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5, color="#1976D2", s=30)
        scale = 2.5
        for i, (val, vec) in enumerate(zip(eigenvalues_pca, eigenvectors_pca.T)):
            col_ev = ["#F44336", "#4CAF50"][i]
            ax1.annotate("", xy=(vec[0]*val*scale/eigenvalues_pca[0], vec[1]*val*scale/eigenvalues_pca[0]),
                         xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color=col_ev, lw=3))
            ax1.text(vec[0]*val*scale/eigenvalues_pca[0]*1.1,
                     vec[1]*val*scale/eigenvalues_pca[0]*1.1,
                     f"PC{i+1}\n(λ={val:.2f})", color=col_ev, fontweight="bold", fontsize=11)
        ax1.set_title("원본 데이터 + 주성분 방향", fontsize=11)
        ax1.grid(True, alpha=0.2); ax1.set_aspect("equal")
        ax1.set_xlabel("x"); ax1.set_ylabel("y")

        projected = data_pca @ eigenvectors_pca[:, 0]
        ax2 = axes_pca[1]
        ax2.hist(projected, bins=20, color="#9C27B0", alpha=0.7, edgecolor="white")
        ax2.set_title(f"1차원으로 투영 (PC1만 사용)\n분산 보존율: {eigenvalues_pca[0]/sum(eigenvalues_pca)*100:.1f}%", fontsize=11)
        ax2.set_xlabel("PC1 방향 값"); ax2.set_ylabel("빈도")
        ax2.grid(True, alpha=0.25)
        plt.tight_layout(); _show(fig_pca)

        col1, col2 = st.columns(2)
        col1.metric("PC1 분산 비율", f"{eigenvalues_pca[0]/sum(eigenvalues_pca)*100:.1f}%")
        col2.metric("PC2 분산 비율", f"{eigenvalues_pca[1]/sum(eigenvalues_pca)*100:.1f}%")
        st.info(f"PC1 방향만으로 데이터의 {eigenvalues_pca[0]/sum(eigenvalues_pca)*100:.1f}%를 설명할 수 있어요!")

    with tab4:
        st.markdown("### 🧩 고유값/고유벡터 퀴즈")
        st.markdown("---")
        quiz("eig1", "Q1. Av = λv 에서 λ(람다)는 무엇인가요?",
             ["행렬", "고유벡터", "고유값", "전치"], 2,
             "λ(람다) = 고유값! 고유벡터의 크기가 얼마나 변하는지를 나타내요.")
        st.markdown("---")
        quiz("eig2", "Q2. PCA에서 '첫 번째 주성분(PC1)'이란?",
             ["평균 방향", "분산이 가장 큰 방향 (최대 고유값의 고유벡터)", "수직 방향", "무작위 방향"], 1,
             "PC1 = 공분산 행렬의 최대 고유값에 대응하는 고유벡터! 데이터가 가장 퍼진 방향이에요.")
        st.markdown("---")
        quiz("eig3", "Q3. 단위 행렬 I의 모든 고유값은?",
             ["0", "1", "-1", "무한대"], 1,
             "Iv = 1×v → 모든 벡터가 고유벡터이고 고유값은 1이에요!")
        st.markdown("---")
        quiz("eig4", "Q4. 고유값 분해(SVD)가 사용되는 AI 응용은?",
             ["경사하강법", "추천 시스템과 이미지 압축", "평균 계산", "확률 계산"], 1,
             "SVD(특이값 분해)는 넷플릭스 추천, 이미지 압축, 자연어 처리에 핵심적으로 사용돼요!")

# ══════════════════════════════════════════════════════════════════════════════
#  AI 튜터
# ══════════════════════════════════════════════════════════════════════════════
def page_ai_tutor():
    st.markdown('<div class="chapter-title">🤖 AI 수학 튜터</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="story-box">
🤖 <b>gemma4 AI 튜터에게 무엇이든 물어보세요!</b><br>
미적분, 통계학, 선형대수학 — 어떤 것도 OK!<br>
각 개념이 AI/딥러닝에 어떻게 쓰이는지도 알려줘요 😊
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
        "수직선이 뭐야?","미분을 피자로 설명해줘",
        "적분이 뭐야?","AI는 미분을 어떻게 쓸까?",
        "평균과 분산의 차이는?","정규분포가 뭔지 쉽게 설명해줘",
        "벡터가 뭐야?","행렬 곱이 AI에서 왜 중요해?",
        "고유값과 고유벡터를 쉽게 설명해줘","PCA가 뭐야?",
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
<div style="font-size:1.15rem;font-weight:900;color:#FFD700;">수학 탐험대</div>
<div style="font-size:.8rem;color:#aaa;">미적분 · 통계 · 선형대수</div>
</div>""", unsafe_allow_html=True)

        selected=st.radio("챕터",list(CHAPTERS.keys()),label_visibility="collapsed")

        st.markdown("---")
        calculus_track=["📏 1. 수직선","📊 2. 함숫값","📐 3. 기울기",
                         "🎢 4. 이차함수","⚡ 5. 미분은 기울기다",
                         "🏞️ 6. 적분은 넓이다","🧠 7. AI와 미적분"]
        stat_track=["📉 8. 평균과 분산","🔔 9. 확률분포","🔗 10. 상관관계와 회귀"]
        la_track=["🏹 11. 벡터","🔲 12. 행렬","🌟 13. 고유값과 고유벡터"]
        all_track=calculus_track+stat_track+la_track
        visited=st.session_state.get("visited_chapters",set())

        st.markdown("#### ➕ 미적분")
        done_c=sum(1 for c in calculus_track if c in visited)
        st.progress(done_c/7,text=f"{done_c}/7 완료")
        for ch in calculus_track:
            st.markdown(f"{'✅' if ch in visited else '⬜'} {ch}")

        st.markdown("#### 📊 통계학")
        done_s=sum(1 for c in stat_track if c in visited)
        st.progress(done_s/3,text=f"{done_s}/3 완료")
        for ch in stat_track:
            st.markdown(f"{'✅' if ch in visited else '⬜'} {ch}")

        st.markdown("#### 🔲 선형대수")
        done_l=sum(1 for c in la_track if c in visited)
        st.progress(done_l/3,text=f"{done_l}/3 완료")
        for ch in la_track:
            st.markdown(f"{'✅' if ch in visited else '⬜'} {ch}")

        st.markdown("---")
        total_done=done_c+done_s+done_l
        st.markdown(f"**전체: {total_done}/13 완료**")
        st.progress(total_done/13)

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
        "stat_descriptive":page_stat_descriptive,
        "stat_distribution":page_stat_distribution,
        "stat_regression":page_stat_regression,
        "linalg_vector":page_linalg_vector,
        "linalg_matrix":page_linalg_matrix,
        "linalg_eigen":page_linalg_eigen,
        "ai_tutor":page_ai_tutor,
    }
    pages[CHAPTERS[selected]]()

if __name__=="__main__":
    main()
