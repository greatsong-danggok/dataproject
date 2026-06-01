import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import io

st.set_page_config(
    page_title="서울 기온 변화 분석",
    page_icon="🌡️",
    layout="wide",
)

# ── 스타일 ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252940);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #30364a;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; margin: 8px 0; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; }
    .before-color { color: #60a5fa; }
    .after-color { color: #f97316; }
    .diff-color { color: #34d399; }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #30364a;
    }
    div[data-testid="stHorizontalBlock"] { gap: 16px; }
</style>
""", unsafe_allow_html=True)

# ── 데이터 로드 ──────────────────────────────────────────
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file, encoding='utf-8-sig')
    else:
        df = pd.read_csv("ta_data.csv", encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['날짜'] = pd.to_datetime(df['날짜'].astype(str).str.strip())
    df['연도'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month
    df['계절'] = df['월'].map(lambda m: '봄' if m in [3,4,5] else ('여름' if m in [6,7,8] else ('가을' if m in [9,10,11] else '겨울')))
    df = df[df['연도'] < 2026]  # 미완성 연도 제외
    return df

# ── 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 설정")
    st.markdown("---")

    uploaded = st.file_uploader("📁 CSV 파일 업로드 (선택)", type="csv",
                                 help="기상청 형식의 CSV를 업로드하면 해당 데이터로 분석합니다.")
    df_raw = load_data(uploaded)

    cutoff_year = st.slider("📅 기준 연도", 1950, 2010, 1980, 5,
                             help="이 연도를 기준으로 '이전/이후'를 구분합니다.")

    season_filter = st.multiselect("🌸 계절 필터", ['봄', '여름', '가을', '겨울'],
                                    default=['봄', '여름', '가을', '겨울'])

    window = st.slider("📈 이동평균 (년)", 1, 20, 10)

    st.markdown("---")
    st.markdown(f"📊 데이터 기간: **{df_raw['연도'].min()}** ~ **{df_raw['연도'].max()}**")
    st.markdown(f"📋 총 데이터: **{len(df_raw):,}** 일")

# ── 데이터 처리 ──────────────────────────────────────────
df = df_raw[df_raw['계절'].isin(season_filter)] if season_filter else df_raw

annual = df.groupby('연도')['평균기온(℃)'].mean().reset_index()
annual.columns = ['연도', '평균기온']
annual['이동평균'] = annual['평균기온'].rolling(window, center=True).mean()

before = annual[annual['연도'] < cutoff_year]['평균기온'].mean()
after = annual[annual['연도'] >= cutoff_year]['평균기온'].mean()
diff = after - before

# 선형 회귀
b_df = annual[annual['연도'] < cutoff_year].dropna(subset=['평균기온'])
a_df = annual[annual['연도'] >= cutoff_year].dropna(subset=['평균기온'])

slope_b, intercept_b, r_b, p_b, _ = stats.linregress(b_df['연도'], b_df['평균기온'])
slope_a, intercept_a, r_a, p_a, _ = stats.linregress(a_df['연도'], a_df['평균기온'])

# ── 헤더 ──────────────────────────────────────────────────
st.markdown("# 🌡️ 서울 기온 변화 분석")
st.markdown(f"**1980년대 전후** 기온 상승 패턴 분석 | 기준 연도: **{cutoff_year}년**")
st.markdown("---")

# ── 요약 지표 ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📘 {cutoff_year}년 이전 평균기온</div>
        <div class="metric-value before-color">{before:.2f}°C</div>
        <div class="metric-label">상승 추세: {slope_b*10:.3f}°C / 10년</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🔴 {cutoff_year}년 이후 평균기온</div>
        <div class="metric-value after-color">{after:.2f}°C</div>
        <div class="metric-label">상승 추세: {slope_a*10:.3f}°C / 10년</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📈 평균기온 차이</div>
        <div class="metric-value diff-color">+{diff:.2f}°C</div>
        <div class="metric-label">이전 대비 {diff/before*100:.1f}% 상승</div>
    </div>""", unsafe_allow_html=True)

with c4:
    speed_ratio = (slope_a / slope_b) if slope_b != 0 else float('inf')
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⚡ 이후/이전 상승속도 비율</div>
        <div class="metric-value" style="color:#a78bfa;">{speed_ratio:.1f}×</div>
        <div class="metric-label">이후 기간 상승 속도가 더 빠름</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 차트 1: 연간 평균기온 + 구분선 ────────────────────────
st.markdown('<div class="section-title">📊 연간 평균기온 추이</div>', unsafe_allow_html=True)

fig1 = go.Figure()

# 배경 음영
fig1.add_vrect(x0=annual['연도'].min()-1, x1=cutoff_year,
               fillcolor="rgba(96,165,250,0.05)", line_width=0, annotation_text="이전 기간",
               annotation_position="top left", annotation_font_color="#60a5fa")
fig1.add_vrect(x0=cutoff_year, x1=annual['연도'].max()+1,
               fillcolor="rgba(249,115,22,0.05)", line_width=0, annotation_text="이후 기간",
               annotation_position="top right", annotation_font_color="#f97316")

# 기준선 (before/after 평균)
fig1.add_hline(y=before, line_dash="dot", line_color="#60a5fa", line_width=1.5,
               annotation_text=f"이전 평균 {before:.2f}°C", annotation_position="right")
fig1.add_hline(y=after, line_dash="dot", line_color="#f97316", line_width=1.5,
               annotation_text=f"이후 평균 {after:.2f}°C", annotation_position="right")

# 연간 데이터
fig1.add_trace(go.Scatter(
    x=annual['연도'], y=annual['평균기온'], mode='lines',
    line=dict(color='rgba(200,200,200,0.3)', width=1),
    name='연간 평균기온', showlegend=True
))

# 이동평균
b_ma = annual[annual['연도'] < cutoff_year]
a_ma = annual[annual['연도'] >= cutoff_year]
fig1.add_trace(go.Scatter(
    x=b_ma['연도'], y=b_ma['이동평균'],
    mode='lines', line=dict(color='#60a5fa', width=2.5),
    name=f'{cutoff_year}년 이전 이동평균({window}년)'
))
fig1.add_trace(go.Scatter(
    x=a_ma['연도'], y=a_ma['이동평균'],
    mode='lines', line=dict(color='#f97316', width=2.5),
    name=f'{cutoff_year}년 이후 이동평균({window}년)'
))

# 추세선
x_b = np.array([b_df['연도'].min(), b_df['연도'].max()])
fig1.add_trace(go.Scatter(
    x=x_b, y=intercept_b + slope_b * x_b,
    mode='lines', line=dict(color='#3b82f6', width=1.5, dash='dash'),
    name=f'이전 추세 ({slope_b*10:+.3f}°C/10년)'
))
x_a = np.array([a_df['연도'].min(), a_df['연도'].max()])
fig1.add_trace(go.Scatter(
    x=x_a, y=intercept_a + slope_a * x_a,
    mode='lines', line=dict(color='#ef4444', width=1.5, dash='dash'),
    name=f'이후 추세 ({slope_a*10:+.3f}°C/10년)'
))

# 기준 연도 수직선
fig1.add_vline(x=cutoff_year, line_dash="solid", line_color="#facc15", line_width=2,
               annotation_text=f"{cutoff_year}년", annotation_font_color="#facc15")

fig1.update_layout(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e5e7eb'),
    height=420,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title='연도'),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title='평균기온 (°C)'),
    margin=dict(l=0, r=80, t=40, b=0),
    hovermode='x unified',
)
st.plotly_chart(fig1, use_container_width=True)

# ── 차트 2: 10년 단위 박스플롯 + 히스토그램 ──────────────
col_l, col_r = st.columns([3, 2])

with col_l:
    st.markdown('<div class="section-title">📦 10년 단위 기온 분포 (박스플롯)</div>', unsafe_allow_html=True)

    df_decade = df.copy()
    df_decade['decade'] = ((df_decade['연도'] // 10) * 10).astype(str) + 's'
    df_decade['기간'] = df_decade['연도'].apply(lambda y: f'이전({cutoff_year}년~)' if y < cutoff_year else f'이후({cutoff_year}년~)')

    decades = sorted(df_decade['decade'].unique())
    colors = ['#60a5fa' if int(d[:-1]) < cutoff_year else '#f97316' for d in decades]

    fig2 = go.Figure()
    for d, c in zip(decades, colors):
        sub = df_decade[df_decade['decade'] == d]['평균기온(℃)'].dropna()
        fig2.add_trace(go.Box(
            y=sub, name=d, marker_color=c,
            line_color=c, fillcolor=c.replace(')', ', 0.3)').replace('rgb', 'rgba') if 'rgb' in c else c,
            showlegend=False, boxmean='sd'
        ))

    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb'), height=350,
        xaxis=dict(showgrid=False, title='10년 단위'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title='일 평균기온 (°C)'),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_r:
    st.markdown('<div class="section-title">📊 이전 vs 이후 기온 분포 비교</div>', unsafe_allow_html=True)

    before_temps = df[df['연도'] < cutoff_year]['평균기온(℃)'].dropna()
    after_temps = df[df['연도'] >= cutoff_year]['평균기온(℃)'].dropna()

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=before_temps, name=f'{cutoff_year}년 이전',
        marker_color='rgba(96,165,250,0.6)', nbinsx=60, histnorm='probability density'
    ))
    fig3.add_trace(go.Histogram(
        x=after_temps, name=f'{cutoff_year}년 이후',
        marker_color='rgba(249,115,22,0.6)', nbinsx=60, histnorm='probability density'
    ))

    # 평균선
    fig3.add_vline(x=before_temps.mean(), line_color='#60a5fa', line_dash='dash', line_width=2)
    fig3.add_vline(x=after_temps.mean(), line_color='#f97316', line_dash='dash', line_width=2)

    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb'), height=350, barmode='overlay',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title='일 평균기온 (°C)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title='밀도'),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── 차트 3: 계절별 기온 변화 ─────────────────────────────
st.markdown('<div class="section-title">🌸 계절별 연간 평균기온 변화</div>', unsafe_allow_html=True)

season_annual = df.groupby(['연도', '계절'])['평균기온(℃)'].mean().reset_index()
season_colors = {'봄': '#34d399', '여름': '#f87171', '가을': '#fbbf24', '겨울': '#818cf8'}

fig4 = make_subplots(rows=2, cols=2,
                      subplot_titles=['🌸 봄 (3-5월)', '☀️ 여름 (6-8월)', '🍂 가을 (9-11월)', '❄️ 겨울 (12-2월)'),
                      vertical_spacing=0.12, horizontal_spacing=0.08)

season_pos = {'봄': (1,1), '여름': (1,2), '가을': (2,1), '겨울': (2,2)}

for s, (r, c) in season_pos.items():
    sub = season_annual[season_annual['계절'] == s].copy()
    sub['ma'] = sub['평균기온(℃)'].rolling(window, center=True).mean()
    color = season_colors[s]

    sub_b = sub[sub['연도'] < cutoff_year]
    sub_a = sub[sub['연도'] >= cutoff_year]

    # 구분 배경
    fig4.add_vrect(x0=sub['연도'].min()-1, x1=cutoff_year,
                   fillcolor="rgba(96,165,250,0.04)", line_width=0, row=r, col=c)
    fig4.add_vrect(x0=cutoff_year, x1=sub['연도'].max()+1,
                   fillcolor="rgba(249,115,22,0.04)", line_width=0, row=r, col=c)
    fig4.add_vline(x=cutoff_year, line_dash="dash", line_color="#facc15",
                   line_width=1, row=r, col=c)

    fig4.add_trace(go.Scatter(
        x=sub['연도'], y=sub['평균기온(℃)'], mode='lines',
        line=dict(color=color.replace(')', ', 0.25)').replace('#', 'rgba(').replace('rgba(', 'rgba(').replace(',', ',').replace('rgba(', 'rgba(0,0,0,') if False else f'rgba(200,200,200,0.2)'),
        showlegend=False, name=s
    ), row=r, col=c)

    if len(sub_b) > 0:
        fig4.add_trace(go.Scatter(
            x=sub_b['연도'], y=sub_b['ma'], mode='lines',
            line=dict(color='#60a5fa', width=2), showlegend=False
        ), row=r, col=c)
    if len(sub_a) > 0:
        fig4.add_trace(go.Scatter(
            x=sub_a['연도'], y=sub_a['ma'], mode='lines',
            line=dict(color='#f97316', width=2), showlegend=False
        ), row=r, col=c)

fig4.update_layout(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e5e7eb', size=11),
    height=450,
    margin=dict(l=0, r=0, t=40, b=0),
)
for i in range(1, 5):
    r, c = [(1,1),(1,2),(2,1),(2,2)][i-1]
    fig4.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)', row=r, col=c)
    fig4.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)', row=r, col=c)

st.plotly_chart(fig4, use_container_width=True)

# ── 통계 검정 ─────────────────────────────────────────────
st.markdown('<div class="section-title">🧪 통계적 유의성 검정</div>', unsafe_allow_html=True)

annual_b = annual[annual['연도'] < cutoff_year]['평균기온'].dropna()
annual_a = annual[annual['연도'] >= cutoff_year]['평균기온'].dropna()

t_stat, p_val = stats.ttest_ind(annual_b, annual_a)
ks_stat, ks_p = stats.ks_2samp(annual_b, annual_a)

# 효과 크기 (Cohen's d)
pooled_std = np.sqrt((annual_b.std()**2 + annual_a.std()**2) / 2)
cohens_d = (annual_a.mean() - annual_b.mean()) / pooled_std

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

def sig_badge(p):
    if p < 0.001: return "🔴 p < 0.001 (매우 유의)"
    elif p < 0.01: return "🟠 p < 0.01 (유의)"
    elif p < 0.05: return "🟡 p < 0.05 (유의)"
    else: return "⚪ p ≥ 0.05 (비유의)"

with stat_col1:
    st.metric("t-검정 통계량", f"{t_stat:.3f}")
    st.caption(sig_badge(p_val))
with stat_col2:
    st.metric("t-검정 p값", f"{p_val:.2e}")
    st.caption("두 기간 평균 차이 검정")
with stat_col3:
    st.metric("KS 검정 p값", f"{ks_p:.2e}")
    st.caption("분포 형태 차이 검정")
with stat_col4:
    effect = "대" if abs(cohens_d) > 0.8 else ("중" if abs(cohens_d) > 0.5 else "소")
    st.metric("Cohen's d (효과 크기)", f"{cohens_d:.3f}")
    st.caption(f"효과 크기: {effect} (|d|={'0.8+' if abs(cohens_d)>0.8 else '0.5+' if abs(cohens_d)>0.5 else '<0.5'})")

# 해석
if p_val < 0.05:
    st.success(f"✅ **{cutoff_year}년을 기준으로 이전과 이후 기간의 평균 기온 차이는 통계적으로 유의합니다** (p = {p_val:.2e}). "
               f"이후 기간이 평균 **+{diff:.2f}°C** 더 높으며, 상승 속도도 **{speed_ratio:.1f}배** 빠릅니다.")
else:
    st.warning(f"⚠️ {cutoff_year}년 기준 분리에서는 통계적 유의성이 낮습니다 (p = {p_val:.2e}). 기준 연도를 조정해 보세요.")

# ── 데이터 테이블 ─────────────────────────────────────────
with st.expander("📋 연간 평균기온 데이터 보기"):
    display_df = annual.copy()
    display_df['기간'] = display_df['연도'].apply(lambda y: f'{cutoff_year}년 이전' if y < cutoff_year else f'{cutoff_year}년 이후')
    display_df['편차 (전체평균 대비)'] = (display_df['평균기온'] - display_df['평균기온'].mean()).round(2)
    st.dataframe(display_df[['연도', '평균기온', '이동평균', '기간', '편차 (전체평균 대비)']].rename(columns={'평균기온':'평균기온(°C)', '이동평균':'이동평균(°C)'}),
                 use_container_width=True, height=300)

    csv_out = display_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button("💾 분석 데이터 다운로드 (CSV)", csv_out, "seoul_temp_analysis.csv", "text/csv")

st.markdown("---")
st.caption("📡 데이터 출처: 기상청 | 지점 108 (서울) | 1907~2025년")
