from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = {'time': 0, 'data': {}, 'macro': {}}
DATA_DIR = './outputs'

try:
    with open(f'{DATA_DIR}/indicator_lookup.json', 'r', encoding='utf-8') as f:
        indicator_lookup = json.load(f)
    with open(f'{DATA_DIR}/markov_lookup.json', 'r', encoding='utf-8') as f:
        markov_lookup = json.load(f)
    with open(f'{DATA_DIR}/thermo_config.json', 'r', encoding='utf-8') as f:
        thermo_config = json.load(f)
    with open(f'{DATA_DIR}/dday_formulas.json', 'r', encoding='utf-8') as f:
        dday_formulas = json.load(f)
    with open(f'{DATA_DIR}/etf_config.json', 'r', encoding='utf-8') as f:
        etf_config = json.load(f)
    print("✅ 5개 통계 데이터 JSON 로드 성공!")
    HAS_STATS = True
except FileNotFoundError:
    print("⚠️ 통계 데이터 JSON을 찾을 수 없습니다. (build_data.py를 먼저 실행하세요)")
    HAS_STATS = False

# ═══════════════════════════════════════
# v7 온도계 가중치 + 투자자 전략 설정
# ═══════════════════════════════════════

THERMO_WEIGHTS = {
    'JPY': {'WR10': 0.45, 'SYNC': 0.15, 'DISP10': 0.20, 'WR7': 0.20},
    'EUR': {'WR10': 0.40, 'WR7': 0.60},
    'THB': {'WR10': 0.45, 'MOM3': 0.45, 'WR14': 0.05, 'STOCH': 0.05},
    'AUD': {'WR10': 1.00},
    'USD': {'WR10': 0.60, 'WR7': 0.15, 'DISP20': 0.05, 'WR20': 0.20},
}

INVESTOR_STRATEGIES = {
    'JPY': {'wr': -95, 'bb_p': 15, 'bb_s': 1.5, 'tp': 1.2, 'sl': 1.5, 'label': 'WR≤-95 + BB(15, 1.5σ)'},
    'EUR': {'wr': -90, 'bb_p': 10, 'bb_s': 1.5, 'tp': 1.0, 'sl': 1.5, 'label': 'WR≤-90 + BB(10, 1.5σ)'},
    'THB': {'wr': -95, 'bb_p': 10, 'bb_s': 1.5, 'tp': 1.0, 'sl': 1.0, 'label': 'WR≤-95 + BB(10, 1.5σ)'},
    'AUD': {'wr': -90, 'bb_p': 10, 'bb_s': 1.5, 'tp': 1.0, 'sl': 1.5, 'label': 'WR≤-90 + BB(10, 1.5σ)'},
    'USD': {'wr': -80, 'bb_p': 20, 'bb_s': 2.5, 'tp': 1.0, 'sl': 1.5, 'label': 'WR≤-80 + BB(20, 2.5σ)'},
}

def get_base_data():
    current_time = time.time()
    if current_time - cache['time'] < 60 and cache['data']:
        return cache['data'], cache['macro']
    
    tickers = {"USD": "KRW=X", "JPY": "JPYKRW=X", "EUR": "EURKRW=X", "AUD": "AUDKRW=X", "THB": "THBKRW=X"}
    raw_data = {}
    for k, v in tickers.items():
        ticker_obj = yf.Ticker(v)
        hist = ticker_obj.history(period="2y")
        # 실시간 현재가 (장중 가격)
        try:
            realtime = float(ticker_obj.info.get('regularMarketPrice', 0))
        except:
            realtime = 0
        raw_data[k] = {
            'Close': hist['Close'].dropna(),
            'High': hist['High'].dropna(),
            'Low': hist['Low'].dropna(),
            'realtime': realtime,
        }
    
    macros = {
        "KOSPI": "^KS11", "SP500": "^GSPC", "DXY": "DX-Y.NYB", 
        "VIX": "^VIX", "EEM": "EEM", "GOLD": "GC=F", 
        "HYG": "HYG", "TLT": "TLT", "NIKKEI": "^N225", "STOXX50": "^STOXX50E"
    }
    macro_res = {}
    for k, v in macros.items():
        d = yf.Ticker(v).history(period="2y")['Close'].dropna()
        if not d.empty:
            val = float(d.iloc[-1])
            p60 = float(d.iloc[-60]) if len(d) >= 60 else float(d.iloc[0])
            ma20 = float(d.rolling(20).mean().iloc[-1]) if len(d) >= 20 else val
            macro_res[k] = {'val': val, 'chg60': float((val-p60)/p60*100), 'ma20': ma20}

    cache['data'] = raw_data
    cache['macro'] = macro_res
    cache['time'] = current_time
    return raw_data, macro_res

def calc_williams_r(high, low, close, period):
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return ((highest - close) / (highest - lowest)) * -100

def calc_all_indicators(data, currency):
    """v7 지표 전부 계산 — score API와 signals API 공용"""
    close = data[currency]['Close']
    high = data[currency]['High']
    low = data[currency]['Low']
    df = pd.DataFrame({'Close': close, 'High': high, 'Low': low})
    
    for p in [7, 10, 14, 20]:
        wr = calc_williams_r(df['High'], df['Low'], df['Close'], p)
        df[f'WR{p}'] = (wr + 100).clip(0, 100)
    
    df['MOM3'] = ((df['Close'].pct_change(3)*100 + 2) / 4 * 100).clip(0, 100)
    
    sync_pool = ["JPY", "EUR", "THB", "AUD"]
    if currency in sync_pool: sync_pool.remove(currency)
    other_moms = []
    for c in sync_pool:
        if c in data:
            c_close = data[c]['Close']
            if len(c_close) >= 6:
                other_moms.append(float(((c_close.iloc[-1] / c_close.iloc[-6]) - 1) * 100))
    df['SYNC'] = float(np.clip(((float(np.mean(other_moms)) + 2) / 4) * 100, 0, 100)) if other_moms else 50
    
    for p in [10, 20]:
        ma = df['Close'].rolling(p).mean()
        df[f'DISP{p}'] = ((df['Close'] / ma - 0.95) / 0.10 * 100).clip(0, 100)
    
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['STOCH'] = ((df['Close'] - low14) / (high14 - low14) * 100).clip(0, 100)
    
    return df

def calc_thermo(df, currency):
    """v7 온도계 점수 + 등급 계산"""
    weights = THERMO_WEIGHTS[currency]
    s = {}
    total = 0
    for ind, weight in weights.items():
        val = float(df[ind].iloc[-1]) if ind in df.columns and not pd.isna(df[ind].iloc[-1]) else 50
        s[ind] = val
        total += val * weight
    
    if total <= 25: grade = "A"
    elif total <= 40: grade = "B"
    elif total <= 55: grade = "C"
    elif total <= 70: grade = "D"
    else: grade = "E"
    
    return total, grade, s

def calc_inv_signal(df, currency, realtime_price=0):
    """v7 투자자 시그널 계산 + 실시간 BB 재확인"""
    strat = INVESTOR_STRATEGIES[currency]
    curr_price = float(df['Close'].iloc[-1])  # 전일 종가
    adj_price = curr_price * (100 if currency == "JPY" else 1)
    curr_wr10 = float(calc_williams_r(df['High'], df['Low'], df['Close'], 10).iloc[-1])
    
    bb_ma = float(df['Close'].rolling(strat['bb_p']).mean().iloc[-1])
    bb_std = float(df['Close'].rolling(strat['bb_p']).std().iloc[-1])
    bb_lower = bb_ma - strat['bb_s'] * bb_std
    bb_lower_display = bb_lower * (100 if currency == "JPY" else 1)
    
    pass_wr = bool(curr_wr10 <= strat['wr'])
    pass_bb = bool(curr_price <= bb_lower)
    inv_signal = pass_wr and pass_bb  # 전일 종가 기준 시그널
    
    # 실시간 BB 재확인
    rt_price = realtime_price if realtime_price > 0 else curr_price
    rt_adj = rt_price * (100 if currency == "JPY" else 1)
    rt_below_bb = bool(rt_price <= bb_lower)
    
    if inv_signal and not rt_below_bb:
        # 어제 시그널 떴지만 오늘 BB 위로 복귀 → 보류
        rt_status = "⚠️ BB 위 복귀 — 매수 보류"
        rt_valid = False
    elif inv_signal and rt_below_bb:
        # 어제 시그널 + 오늘도 BB 아래 → 유효!
        rt_status = "✅ 실시간 확인 — 매수 유효!"
        rt_valid = True
    else:
        rt_status = ""
        rt_valid = False
    
    tp_price = round(rt_adj * (1 + strat['tp']/100), 2)
    sl_price = round(rt_adj * (1 - strat['sl']/100), 2)
    
    inv_conds = [
        {"name": "WR(10) 전일종가", "target": f"≤ {strat['wr']}", "current": f"{curr_wr10:.1f}", "pass": pass_wr},
        {"name": f"BB 하단({strat['bb_p']}, {strat['bb_s']}σ)", "target": f"≤ {bb_lower_display:.2f}", "current": f"{adj_price:.2f}", "pass": pass_bb},
    ]
    
    # 실시간 가격 조건 추가 (시그널 뜬 경우만)
    if inv_signal:
        inv_conds.append({
            "name": "📡 실시간 BB 재확인",
            "target": f"≤ {bb_lower_display:.2f}",
            "current": f"{rt_adj:.2f}",
            "pass": rt_below_bb,
        })
    
    inv_tpsl = f"익절 {strat['tp']}% / 손절 {strat['sl']}% / 타임아웃 20일"
    if inv_signal and rt_valid:
        inv_tpsl += f" | 💰 익절가: {tp_price:.2f} | 🛑 손절가: {sl_price:.2f}"
    
    return inv_signal, strat['label'], inv_tpsl, inv_conds, curr_wr10, tp_price, sl_price, rt_status, rt_valid

@app.get("/api/score")
def get_score(currency: str = "USD", d_day: int = 14, mode: str = "traveler"):
    try:
        data, macro = get_base_data()
        df = calc_all_indicators(data, currency)
        
        curr_price = float(df['Close'].iloc[-1])
        adj_price = curr_price * (100 if currency == "JPY" else 1)
        
        # 온도계
        total, grade, s = calc_thermo(df, currency)
        
        # D-Day 완화 (기존 구조 유지)
        threshold = 50
        if HAS_STATS:
            formula = dday_formulas.get(currency, {})
            for d in sorted([int(k) for k in formula.keys()], reverse=True):
                if d_day >= d:
                    threshold = formula[str(d)]
                    break

        # 투자자 시그널
        rt_price = data[currency].get('realtime', 0)
        inv_signal, inv_strategy, inv_tpsl, inv_conds, curr_wr10, tp_price, sl_price, rt_status, rt_valid = calc_inv_signal(df, currency, rt_price)

        # ═══════════════════════════════════════
        # 거시경고 (기존 구조 그대로)
        # ═══════════════════════════════════════
        
        m_items = []; w_cnt = []; active_warnings = []
        
        k_chg = macro.get('KOSPI', {}).get('chg60', 0)
        sp_chg = macro.get('SP500', {}).get('chg60', 0)
        hyg_chg = macro.get('HYG', {}).get('chg60', 0)
        eem_chg = macro.get('EEM', {}).get('chg60', 0)
        dxy_chg = macro.get('DXY', {}).get('chg60', 0)
        vix_val = macro.get('VIX', {}).get('val', 0)
        gold_chg = macro.get('GOLD', {}).get('chg60', 0)
        tlt_chg = macro.get('TLT', {}).get('chg60', 0)
        nikkei_chg = macro.get('NIKKEI', {}).get('chg60', 0)
        stoxx_chg = macro.get('STOXX50', {}).get('chg60', 0)

        def add_macro(name, val, is_pct, limit, is_less_than, msg):
            is_warn = (val < limit) if is_less_than else (val > limit)
            if is_warn:
                w_cnt.append(1)
                active_warnings.append(f"🔴 {msg}")
            v_str = f"{val:+.1f}%" if is_pct else f"{val:.1f}"
            c_str = f"위험: {'<' if is_less_than else '>'} {limit}{'%' if is_pct else ''}"
            m_items.append({"n": name, "c": c_str, "v": v_str, "w": bool(is_warn)})

        add_macro("KOSPI 60일", k_chg, True, -10, True, "KOSPI 하락 (한국 증시 폭락)")
        add_macro("S&P500 60일", sp_chg, True, -10, True, "S&P500 하락 (미국 증시 폭락)")

        if currency == "JPY":
            add_macro("HYG 60일", hyg_chg, True, -5, True, "하이일드(HYG) 하락")
            add_macro("EEM 60일", eem_chg, True, -10, True, "신흥국(EEM) 자금 이탈")
            add_macro("달러(DXY) 60일", dxy_chg, True, 5, False, "DXY 강세 (달러 모멘텀)")
            add_macro("닛케이 60일", nikkei_chg, True, -10, True, "닛케이 하락 (안전선호)")
        elif currency == "EUR":
            add_macro("STOXX50 60일", stoxx_chg, True, -10, True, "유로STOXX50 하락")
            add_macro("EEM 60일", eem_chg, True, -10, True, "신흥국(EEM) 자금 이탈")
        elif currency == "THB":
            add_macro("달러(DXY) 60일", dxy_chg, True, 3, False, "DXY 강세 (달러 모멘텀)")
            add_macro("EEM 60일", eem_chg, True, -10, True, "신흥국(EEM) 자금 이탈")
        elif currency == "AUD":
            add_macro("VIX", vix_val, False, 25, False, "VIX 급등 (글로벌 공포)")
            add_macro("HYG 60일", hyg_chg, True, -5, True, "하이일드(HYG) 하락")
            add_macro("EEM 60일", eem_chg, True, -10, True, "신흥국(EEM) 자금 이탈")
            add_macro("금(GOLD) 60일", gold_chg, True, 15, False, "GOLD 급등 (안전자산 수요)")
            add_macro("미국채(TLT) 60일", tlt_chg, True, 10, False, "TLT 급등 (안전선호)")
        elif currency == "USD":
            add_macro("EEM 60일", eem_chg, True, -10, True, "신흥국(EEM) 자금 이탈")

        warn_total = sum(w_cnt)
        capped_w_cnt = min(warn_total, 5)

        # 투자 가이드 — 통화별 분리 (v7)
        if HAS_STATS:
            currency_guides = etf_config.get('investor_guides', {}).get(currency, {})
            max_key = str(max(int(k) for k in currency_guides.keys())) if currency_guides else "0"
            gd = currency_guides.get(str(capped_w_cnt), currency_guides.get(max_key, {"fut": "N/A", "down": "N/A", "holder": "확인 필요", "buyer": "확인 필요"}))
        else:
            gd = {"fut": "N/A", "down": "N/A", "holder": "데이터 없음", "buyer": "데이터 없음"}

        # 개별 지표 상세 (기존 구조 유지)
        sub_details_dict = {}
        for k, v in s.items():
            score_int = int(v)
            sub_details_dict[k] = {"score": score_int}
            if HAS_STATS:
                ind_data = indicator_lookup.get(currency, {}).get(k, {}).get('scores', {}).get(str(score_int))
                if ind_data:
                    sub_details_dict[k].update({"grade": ind_data.get("grade", ""), "win": ind_data.get("win", 0), "saving": ind_data.get("saving", 0.0)})

        # AI 메시지 — 모드별 분기
        stat_msg = ""
        stat_details = {}
        if HAS_STATS:
            str_total = str(int(total))
            thermo_data = indicator_lookup.get(currency, {}).get('THERMO', {}).get('scores', {}).get(str_total)
            if thermo_data:
                stat_details['win_rate'] = thermo_data['win']
                stat_details['saving'] = thermo_data['saving']
            
            # ═══════════════════════════════════════
            # 투자자 모드: 백테스트 통계만 표시
            # ═══════════════════════════════════════
            if mode == "investor":
                # 통화별 백테스트 결과 (CAGR 포함)
                investor_stats = {
                    'USD': {'win': 86.4, 'cagr': 5.22, 'trades': 3, 'avg_hold': 4.0},
                    'JPY': {'win': 81.8, 'cagr': 9.28, 'trades': 7, 'avg_hold': 4.3},
                    'EUR': {'win': 81.6, 'cagr': 12.05, 'trades': 10, 'avg_hold': 4.0},
                    'AUD': {'win': 85.1, 'cagr': 12.77, 'trades': 9, 'avg_hold': 3.5},
                    'THB': {'win': 87.5, 'cagr': 10.35, 'trades': 7, 'avg_hold': 2.8},
                }
                
                stats = investor_stats.get(currency, {'win': 0, 'cagr': 0, 'trades': 0, 'avg_hold': 0})
                
                stat_msg = f"<div style='padding:12px; background:var(--card-bg); border-radius:8px; border:1px solid var(--border-color);'>"
                stat_msg += f"🌍 <strong>시장 경보 Level {warn_total}</strong><br>"
                if warn_total > 0:
                    stat_msg += f"<span style='color:#E24B4A; font-size:12px;'>" + "<br>".join(active_warnings) + "</span>"
                    stat_msg += f"<hr style='margin:8px 0; border:none; border-top:1px dashed var(--border-color);'>"
                    stat_msg += f"<span style='color:#E24B4A; font-size:12px;'>⚠️ <b>거시 경고 발동 — 시그널 발생해도 신중 진입</b></span>"
                else:
                    stat_msg += f"<span style='color:#1D9E75; font-size:12px;'>🟢 거시 환경 안정</span>"
                stat_msg += f"</div>"
                
                stat_msg += f"<br><div style='padding:12px; background:var(--card-bg); border-radius:8px; border:1px solid var(--border-color);'>"
                stat_msg += f"<span style='font-size:13px; font-weight:700;'>📊 검증 결과 (20년 백테스트)</span><br>"
                stat_msg += f"<div style='font-size:12px; margin-top:6px; line-height:1.8;'>"
                stat_msg += f"• 승률: <strong>{stats['win']:.1f}%</strong><br>"
                stat_msg += f"• CAGR: <strong style='color:#1D9E75'>+{stats['cagr']:.2f}%</strong> <span style='font-size:10px; color:var(--text-sub)'>(연평균 수익률, 100만원 기준)</span><br>"
                stat_msg += f"• 연 평균 거래: <strong>{stats['trades']}회</strong><br>"
                stat_msg += f"• 평균 보유: <strong>{stats['avg_hold']:.1f}일</strong>"
                stat_msg += f"</div></div>"
                
            # ═══════════════════════════════════════
            # 여행자 모드: 온도계 기반 가이드
            # ═══════════════════════════════════════
            else:
                if thermo_data:
                    stat_msg = f"과거 통계상 이 구간 진입 시 승률은 <strong>{thermo_data['win']}%</strong> 였습니다."
                
                markov_7d = markov_lookup.get(currency, {}).get('7', {}).get(str_total)
                if markov_7d:
                    stat_msg += f"<br>7일 뒤 하락(유리)할 확률: <strong>{markov_7d.get('down', markov_7d.get('cheaper', 'N/A'))}%</strong>"
                
                stat_msg += f"<br><br><div style='padding:12px; background:var(--card-bg); border-radius:8px; border:1px solid var(--border-color);'>"
                stat_msg += f"🌍 <strong>시장 경보 Level {warn_total}</strong><br>"
                if warn_total > 0:
                    stat_msg += f"<span style='color:#E24B4A; font-size:12px;'>" + "<br>".join(active_warnings) + "</span><hr style='margin:8px 0; border:none; border-top:1px dashed var(--border-color);'>"
                else:
                    stat_msg += f"<span style='color:#1D9E75; font-size:12px;'>🟢 모든 통화 맞춤형 거시지표가 안정적입니다.</span><hr style='margin:8px 0; border:none; border-top:1px dashed var(--border-color);'>"
                
                # 등급 + 60일 추세 조합 메시지 (여행자 모드 전용)
                fut_str = gd.get('fut', '+0.0%')
                try:
                    fut_val = float(fut_str.replace('%', '').replace('+', ''))
                except:
                    fut_val = 0
                
                # 단기 등급 해석
                if grade == 'A':
                    short_term = "🟢 <b>지금 매수 적기!</b> (단기 매우 쌈)"
                    short_color = "#1D9E75"
                elif grade == 'B':
                    short_term = "🟢 단기 양호한 가격"
                    short_color = "#1D9E75"
                elif grade == 'C':
                    short_term = "⚪ 평범한 가격"
                    short_color = "var(--text-main)"
                elif grade == 'D':
                    short_term = "🟡 단기 약간 비쌈"
                    short_color = "#F6A800"
                else:  # E
                    short_term = "🔴 <b>단기 비쌈</b>"
                    short_color = "#E24B4A"
                
                # 60일 추세 해석
                if fut_val >= 3:
                    trend_text = f"📈 <b>강한 상승 추세</b> (60일 후 +{fut_val:.1f}%)"
                    trend_color = "#E24B4A"
                elif fut_val >= 1:
                    trend_text = f"📈 약상승 추세 (60일 후 +{fut_val:.1f}%)"
                    trend_color = "#F6A800"
                elif fut_val >= -1:
                    trend_text = f"⚪ 횡보 추세 (60일 후 {fut_val:+.1f}%)"
                    trend_color = "var(--text-main)"
                else:
                    trend_text = f"📉 하락 추세 (60일 후 {fut_val:+.1f}%)"
                    trend_color = "#1D9E75"
                
                # 종합 판단
                if grade in ['A', 'B']:
                    if fut_val >= 1:
                        verdict = "🎯 <b>지금 환전 강력 추천!</b> (단기 싸고 + 장기 상승)"
                        verdict_color = "#1D9E75"
                    elif fut_val >= -1:
                        verdict = "✅ <b>지금 환전 OK</b>"
                        verdict_color = "#1D9E75"
                    else:
                        verdict = "🤔 단기 좋지만 장기 하락 — 출국 임박시 환전"
                        verdict_color = "var(--text-main)"
                elif grade == 'C':
                    if fut_val >= 2:
                        verdict = "⚡ 평범하지만 <b>장기 상승</b> — 더 비싸지기 전 환전"
                        verdict_color = "#F6A800"
                    elif fut_val >= -1:
                        verdict = "⚪ 평범 — 본인 일정에 맞춰 환전"
                        verdict_color = "var(--text-main)"
                    else:
                        verdict = "⏳ 평범하지만 장기 하락 — 1~2주 대기 가능"
                        verdict_color = "#1D9E75"
                else:  # D, E
                    if fut_val >= 2:
                        verdict = f"⚠️ <b>단기 비싸지만 추세 상승 — 더 비싸질 확률 높음</b>"
                        verdict_color = "#F6A800"
                    elif fut_val >= 0:
                        verdict = "⏳ 단기 비쌈 + 추세 미약 — <b>1~2주 대기 추천</b>"
                        verdict_color = "#1D9E75"
                    else:
                        verdict = "✅ <b>대기 권장!</b> (단기 비싸고 + 장기 하락)"
                        verdict_color = "#1D9E75"
                
                stat_msg += f"<span style='font-size:13px; font-weight:700;'>💡 종합 판단</span><br>"
                stat_msg += f"<div style='padding:8px; background:rgba(0,0,0,0.03); border-radius:6px; margin:4px 0;'>"
                stat_msg += f"<span style='font-size:12px;'>"
                stat_msg += f"단기 ({grade}등급): <span style='color:{short_color}'>{short_term}</span><br>"
                stat_msg += f"장기 (60일): <span style='color:{trend_color}'>{trend_text}</span>"
                stat_msg += f"</span></div>"
                stat_msg += f"<div style='padding:10px; background:{verdict_color}15; border-left:3px solid {verdict_color}; border-radius:6px; margin-top:6px;'>"
                stat_msg += f"<span style='font-size:13px; color:{verdict_color}'>{verdict}</span>"
                stat_msg += f"</div>"
                
                stat_msg += "</div>"

        return {
            "price": adj_price, "rsi": int(total), "threshold": int(threshold), "signal": bool(total <= threshold),
            "grade": grade, "is_investable": True, "inv_signal": bool(inv_signal),
            "inv_strategy": inv_strategy, "inv_tpsl": inv_tpsl, "inv_conds": inv_conds, "sub_details": sub_details_dict,
            "macro_items": m_items, "macro_count": warn_total, "ai_message": stat_msg, "stat_details": stat_details,
            "rt_status": rt_status, "rt_valid": rt_valid,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"price": 0, "rsi": 50, "threshold": 50, "signal": False, "grade": "C", "is_investable": False, "inv_signal": False, "inv_strategy": "", "inv_tpsl": "", "inv_conds": [], "sub_details": {}, "macro_items": [], "macro_count": 0, "ai_message": "데이터 오류", "stat_details": {}}

@app.get("/api/signals")
def get_all_signals():
    """전 통화 시그널 한번에 체크 — 상단 배너용"""
    try:
        data, macro = get_base_data()
        signals = []
        
        for currency in ["USD", "JPY", "EUR", "AUD", "THB"]:
            df = calc_all_indicators(data, currency)
            curr_price = float(df['Close'].iloc[-1])
            adj_price = curr_price * (100 if currency == "JPY" else 1)
            
            total, grade, s = calc_thermo(df, currency)
            rt_price = data[currency].get('realtime', 0)
            inv_signal, strategy, _, _, wr10, tp_price, sl_price, rt_status, rt_valid = calc_inv_signal(df, currency, rt_price)
            
            signals.append({
                "currency": currency,
                "price": adj_price,
                "thermo": round(total, 1),
                "grade": grade,
                "wr10": round(wr10, 1),
                "inv_signal": inv_signal,
                "rt_valid": rt_valid,
                "rt_status": rt_status,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "strategy": strategy,
            })
        
        return {"signals": signals}
    except Exception as e:
        print(f"Signals Error: {e}")
        return {"signals": []}

@app.get("/api/chart")
def get_chart(currency: str = "USD", days: int = 90):
    """일별 차트 + 볼린저밴드 + WR 데이터"""
    try:
        data, macro = get_base_data()
        close = data[currency]['Close']
        high = data[currency]['High']
        low = data[currency]['Low']
        
        df = pd.DataFrame({'Close': close, 'High': high, 'Low': low})
        
        # WR10 (차트 하단 표시용)
        df['WR10'] = calc_williams_r(df['High'], df['Low'], df['Close'], 10)
        
        # BB (투자자 전략에 맞는 BB)
        strat = INVESTOR_STRATEGIES[currency]
        bb_period = strat['bb_p']
        bb_sigma = strat['bb_s']
        
        ma = df['Close'].rolling(bb_period).mean()
        std = df['Close'].rolling(bb_period).std()
        df['MA'] = ma
        df['BB_Upper'] = ma + bb_sigma * std
        df['BB_Lower'] = ma - bb_sigma * std
        
        df = df.tail(days)
        df.dropna(inplace=True)
        
        mult = 100 if currency == "JPY" else 1
        chart_data = []
        for date, row in df.iterrows():
            chart_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "close": round(float(row['Close']) * mult, 2),
                "ma": round(float(row['MA']) * mult, 2),
                "bb_upper": round(float(row['BB_Upper']) * mult, 2),
                "bb_lower": round(float(row['BB_Lower']) * mult, 2),
                "rsi": round(float(row['WR10']), 1) if not pd.isna(row['WR10']) else -50
            })
        
        return {
            "currency": currency,
            "bb_period": bb_period,
            "bb_sigma": bb_sigma,
            "data": chart_data
        }
    except Exception as e:
        print(f"Chart Error: {e}")
        return {"currency": currency, "data": []}

@app.get("/")
def home():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "index.html")
    return FileResponse(html_path)