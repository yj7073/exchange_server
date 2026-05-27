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

import threading

# 백그라운드 워커가 채우는 데이터 저장소 (사용자 요청은 여기서만 읽음)
_data_store = {
    'data': {},      # 환율 5개
    'macro': {},     # 매크로 10개
    'last_update': 0,
    'ready': False,  # 첫 데이터 로드 완료 여부
}
_lock = threading.Lock()
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
    'THB': {'DISP10': 0.60, 'WR10': 0.40},
    'AUD': {'WR10': 1.00},
    'USD': {'WR7': 0.60, 'RSI14': 0.40},
}

# 통화별 ABCDF 등급 경계 (그리드서치 최적화) — A≤[0] B≤[1] C≤[2] D≤[3] F>[3]
GRADE_CUTS = {
    'EUR': (15, 36, 60, 90),
    'JPY': (22, 36, 52, 76),
    'THB': (32, 43, 56, 72),
    'AUD': (18, 38, 62, 90),
    'USD': (26, 40, 56, 76),
}

# ═══════════════════════════════════════
# 검증된 투자 전략 (walk-forward / 그리드서치 / OOS 검증 기반)
#   sig: 신호 지표 (RSI 또는 WR)  |  period: 지표 기간
#   th: 신호 임계값  |  bb_p/bb_s: 볼린저밴드 기간/시그마
#   tp/sl: 익절/손절 %  |  to: 타임아웃(거래일)
#   gap: 상승갭 컷 기준 % (하락갭은 컷 안 함 — 비대칭)
#   vix: VIX 필터 상한 (None=미적용)  |  grade: 강건성 등급
#   investable: 투자신호 제공 여부 (EUR은 검증 결과 부적합)
# ═══════════════════════════════════════
INVESTOR_STRATEGIES = {
    'USD': {'sig': 'RSI', 'period': 7,  'th': 35,  'bb_p': 13, 'bb_s': 1.5,
            'tp': 0.8, 'sl': 1.5, 'to': 20, 'gap': 0.5, 'vix': 25,
            'grade': '★★★', 'investable': True,
            'label': 'RSI(7)≤35 + BB(13, 1.5σ)'},
    'JPY': {'sig': 'WR',  'period': 21, 'th': -95, 'bb_p': 20, 'bb_s': 1.5,
            'tp': 0.8, 'sl': 1.5, 'to': 15, 'gap': 0.5, 'vix': 25,
            'grade': '★★', 'investable': True,
            'label': 'WR(21)≤-95 + BB(20, 1.5σ)'},
    'AUD': {'sig': 'WR',  'period': 14, 'th': -80, 'bb_p': 13, 'bb_s': 1.5,
            'tp': 1.2, 'sl': 0.8, 'to': 15, 'gap': 0.5, 'vix': None,
            'grade': '★★', 'investable': True,
            'label': 'WR(14)≤-80 + BB(13, 1.5σ)'},
    'THB': {'sig': 'RSI', 'period': 7,  'th': 45,  'bb_p': 13, 'bb_s': 1.5,
            'tp': 1.0, 'sl': 1.0, 'to': 20, 'gap': 0.5, 'vix': None,
            'grade': '★', 'investable': True,
            'label': 'RSI(7)≤45 + BB(13, 1.5σ)'},
    'EUR': {'sig': 'WR',  'period': 14, 'th': -80, 'bb_p': 13, 'bb_s': 1.5,
            'tp': 1.0, 'sl': 1.5, 'to': 15, 'gap': 0.5, 'vix': None,
            'grade': '—', 'investable': False,
            'label': '검증 결과 평균회귀 부적합'},
}

def fetch_from_yfinance():
    """
    yfinance에서 실제 데이터를 가져오는 함수.
    백그라운드 워커가 호출하며, 사용자 요청 경로에서는 호출되지 않음.
    각 통화/매크로 호출이 실패해도 이전 데이터를 유지하여 안정성 보장.
    """
    tickers = {"USD": "KRW=X", "JPY": "JPYKRW=X", "EUR": "EURKRW=X", "AUD": "AUDKRW=X", "THB": "THBKRW=X"}
    raw_data = {}
    for k, v in tickers.items():
        try:
            ticker_obj = yf.Ticker(v)
            hist = ticker_obj.history(period="2y")
            # 일봉 마지막 종가 (안전 기본값)
            current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
            # 실시간 현재가: 1일치 .history() 사용 (빠르고 안정적)
            try:
                rt_hist = ticker_obj.history(period="1d")
                if not rt_hist.empty:
                    realtime = float(rt_hist['Close'].iloc[-1])
                else:
                    realtime = current_price
            except Exception as e:
                print(f"⚠️ {k} 실시간 가격 로드 실패: {e}")
                realtime = current_price
            raw_data[k] = {
                'Close': hist['Close'].dropna(),
                'High': hist['High'].dropna(),
                'Low': hist['Low'].dropna(),
                'realtime': realtime,
            }
        except Exception as e:
            # 한 통화 실패해도 이전 데이터 유지
            print(f"⚠️ {k} yfinance 호출 실패: {e}")
            if k in _data_store.get('data', {}):
                raw_data[k] = _data_store['data'][k]
    
    macros = {
        "KOSPI": "^KS11", "SP500": "^GSPC", "DXY": "DX-Y.NYB", 
        "VIX": "^VIX", "EEM": "EEM", "GOLD": "GC=F", 
        "HYG": "HYG", "TLT": "TLT", "NIKKEI": "^N225", "STOXX50": "^STOXX50E"
    }
    macro_res = {}
    for k, v in macros.items():
        try:
            d = yf.Ticker(v).history(period="2y")['Close'].dropna()
            if not d.empty:
                val = float(d.iloc[-1])
                p60 = float(d.iloc[-60]) if len(d) >= 60 else float(d.iloc[0])
                ma20 = float(d.rolling(20).mean().iloc[-1]) if len(d) >= 20 else val
                macro_res[k] = {'val': val, 'chg60': float((val-p60)/p60*100), 'ma20': ma20}
        except Exception as e:
            print(f"⚠️ {k} 매크로 로드 실패: {e}")
            if k in _data_store.get('macro', {}):
                macro_res[k] = _data_store['macro'][k]
    
    return raw_data, macro_res


def background_updater():
    """
    1분마다 백그라운드에서 yfinance 데이터 갱신.
    실패해도 이전 데이터를 유지하여 사용자에게 영향 없음.
    """
    while True:
        time.sleep(60)  # 1분 대기
        try:
            new_data, new_macro = fetch_from_yfinance()
            with _lock:
                _data_store['data'] = new_data
                _data_store['macro'] = new_macro
                _data_store['last_update'] = time.time()
            print(f"✅ 백그라운드 갱신 완료: {time.strftime('%H:%M:%S')}")
        except Exception as e:
            # 실패해도 이전 데이터 유지, 앱 계속 실행
            print(f"⚠️ 백그라운드 갱신 실패 (이전 데이터 유지): {e}")


def get_base_data():
    """
    사용자 요청용 — 메모리에서만 읽어 즉시 반환 (~5ms).
    첫 호출 시에만 yfinance 호출 (앱 시작 후 첫 요청).
    """
    # 초기화 안 됐으면 강제 로드 (락 밖에서)
    if not _data_store['ready']:
        try:
            data, macro = fetch_from_yfinance()
            with _lock:
                _data_store['data'] = data
                _data_store['macro'] = macro
                _data_store['ready'] = True
                _data_store['last_update'] = time.time()
        except Exception as e:
            print(f"❌ 초기 데이터 로드 실패: {e}")
            return {}, {}
    
    # 메모리 데이터 반환 (락으로 일관성 보장)
    with _lock:
        return _data_store['data'], _data_store['macro']

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
    
    for p in [7, 14]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
        rs = gain / (loss + 1e-10)
        df[f'RSI{p}'] = (100 - (100 / (1 + rs))).clip(0, 100)
    
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
    
    a, b, c, d = GRADE_CUTS[currency]
    if total <= a: grade = "A"
    elif total <= b: grade = "B"
    elif total <= c: grade = "C"
    elif total <= d: grade = "D"
    else: grade = "F"
    
    return total, grade, s

def _calc_rsi(close, period):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return (100 - (100 / (1 + rs)))

def calc_inv_signal(df, currency, realtime_price=0, vix_val=0):
    """검증된 투자 시그널 — 통화별 RSI/WR + 갭필터(비대칭) + VIX 필터"""
    strat = INVESTOR_STRATEGIES[currency]
    is_jpy = (currency == "JPY")
    px100 = (100 if is_jpy else 1)

    curr_price = float(df['Close'].iloc[-1])   # 전일 종가(신호 기준가)
    adj_price = curr_price * px100

    # ── EUR: 검증 결과 투자 부적합 → 신호 미제공 ──
    if not strat.get('investable', True):
        msg = "⚠️ 검증 결과 평균회귀 전략 부적합 — 투자 신호를 제공하지 않습니다."
        info = [{"name": "검증 결과", "target": "—",
                 "current": "IS Sharpe 음수 / walk-forward 43%", "pass": False}]
        tpsl = "이 통화는 투자 신호 대상이 아닙니다 (환율 정보·온도계는 이용 가능)"
        return False, strat['label'], tpsl, info, 0.0, 0.0, 0.0, msg, False

    # ── 통화별 신호 지표 계산 ──
    if strat['sig'] == 'RSI':
        ind_series = _calc_rsi(df['Close'], strat['period'])
        ind_val = float(ind_series.iloc[-1])
        pass_sig = bool(ind_val <= strat['th'])
        sig_name = f"RSI({strat['period']}) 전일종가"
        sig_target = f"≤ {strat['th']}"
        sig_cur = f"{ind_val:.1f}"
    else:  # WR (원본 스케일 -100~0)
        ind_series = calc_williams_r(df['High'], df['Low'], df['Close'], strat['period'])
        ind_val = float(ind_series.iloc[-1])
        pass_sig = bool(ind_val <= strat['th'])
        sig_name = f"WR({strat['period']}) 전일종가"
        sig_target = f"≤ {strat['th']}"
        sig_cur = f"{ind_val:.1f}"

    # ── 볼린저밴드 하단 ──
    bb_ma = float(df['Close'].rolling(strat['bb_p']).mean().iloc[-1])
    bb_std = float(df['Close'].rolling(strat['bb_p']).std().iloc[-1])
    bb_lower = bb_ma - strat['bb_s'] * bb_std
    bb_lower_display = bb_lower * px100
    pass_bb = bool(curr_price <= bb_lower)

    # ── VIX 필터 (USD/JPY) ──
    pass_vix = True
    vix_cond = None
    if strat['vix'] is not None:
        pass_vix = bool(vix_val <= strat['vix']) if vix_val > 0 else True
        vix_cond = {"name": f"VIX ≤ {strat['vix']} (위기 회피)",
                    "target": f"≤ {strat['vix']}",
                    "current": f"{vix_val:.1f}" if vix_val > 0 else "N/A",
                    "pass": pass_vix}

    inv_signal = pass_sig and pass_bb and pass_vix  # 전일 종가 기준 시그널

    # ── 갭필터 (비대칭): 상승 +gap% 컷 / 하락은 통과 ──
    rt_price = realtime_price if realtime_price > 0 else curr_price
    rt_adj = rt_price * px100
    gap_pct = (rt_price - curr_price) / curr_price * 100 if curr_price else 0
    gap_up_cut = strat['gap']

    if inv_signal and gap_pct >= gap_up_cut:
        rt_status = f"⚠️ 갭 상승 {gap_pct:+.2f}% — 매수 보류 (이미 +{gap_up_cut}% 넘게 오름)"
        rt_valid = False
    elif inv_signal and gap_pct < 0:
        rt_status = f"✅ 매수 가능 (갭 {gap_pct:+.2f}% — 더 좋은 가격)"
        rt_valid = True
    elif inv_signal:
        rt_status = f"✅ 매수 가능 (갭 {gap_pct:+.2f}% — +{gap_up_cut}% 이내)"
        rt_valid = True
    else:
        rt_status = ""
        rt_valid = False

    # 매수가(실시간) 기준 익절/손절가
    tp_price = round(rt_adj * (1 + strat['tp']/100), 2)
    sl_price = round(rt_adj * (1 - strat['sl']/100), 2)

    inv_conds = [
        {"name": sig_name, "target": sig_target, "current": sig_cur, "pass": pass_sig},
        {"name": f"BB 하단({strat['bb_p']}, {strat['bb_s']}σ)",
         "target": f"≤ {bb_lower_display:.2f}", "current": f"{adj_price:.2f}", "pass": pass_bb},
    ]
    if vix_cond is not None:
        inv_conds.append(vix_cond)

    # 갭 확인 항목 (시그널 뜬 경우만)
    if inv_signal:
        inv_conds.append({
            "name": "📡 갭 확인 (상승 +0.5% 컷 / 하락 OK)",
            "target": f"< +{gap_up_cut}%",
            "current": f"{gap_pct:+.2f}%",
            "pass": rt_valid,
        })

    inv_tpsl = f"익절 {strat['tp']}% / 손절 {strat['sl']}% / 타임아웃 {strat['to']}일"
    if inv_signal and rt_valid:
        inv_tpsl += f" | 💰 익절가: {tp_price:.2f} | 🛑 손절가: {sl_price:.2f}"

    return inv_signal, strat['label'], inv_tpsl, inv_conds, ind_val, tp_price, sl_price, rt_status, rt_valid

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

        # 투자자 시그널 (VIX 필터용 값 먼저 추출)
        rt_price = data[currency].get('realtime', 0)
        _vix_for_signal = macro.get('VIX', {}).get('val', 0)
        inv_signal, inv_strategy, inv_tpsl, inv_conds, curr_wr10, tp_price, sl_price, rt_status, rt_valid = calc_inv_signal(df, currency, rt_price, _vix_for_signal)

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

        def add_macro(name, val, is_pct, limit, is_less_than, msg, help_key):
            is_warn = (val < limit) if is_less_than else (val > limit)
            if is_warn:
                w_cnt.append(1)
                # 형식: "🔴 메시지|||도움말키" — 프론트에서 ⓘ 아이콘으로 변환
                active_warnings.append(f"🔴 {msg}|||{help_key}")
            v_str = f"{val:+.1f}%" if is_pct else f"{val:.1f}"
            c_str = f"위험: {'<' if is_less_than else '>'} {limit}{'%' if is_pct else ''}"
            m_items.append({"n": name, "c": c_str, "v": v_str, "w": bool(is_warn), "h": help_key})

        add_macro("KOSPI 60일", k_chg, True, -10, True, "KOSPI 하락 (한국 증시 폭락)", "KOSPI")
        add_macro("S&P500 60일", sp_chg, True, -10, True, "S&P500 하락 (미국 증시 폭락)", "SP500")

        if currency == "JPY":
            add_macro("HYG 60일", hyg_chg, True, -5, True, "하이일드(HYG) 하락", "HYG")
            add_macro("EEM 60일", eem_chg, True, -10, True, "신흥국(EEM) 자금 이탈", "EEM")
            add_macro("달러(DXY) 60일", dxy_chg, True, 5, False, "DXY 강세 (달러 모멘텀)", "DXY")
            add_macro("닛케이 60일", nikkei_chg, True, -10, True, "닛케이 하락 (안전선호)", "NIKKEI")
        elif currency == "EUR":
            add_macro("STOXX50 60일", stoxx_chg, True, -10, True, "유로STOXX50 하락", "STOXX50")
            add_macro("EEM 60일", eem_chg, True, -10, True, "신흥국(EEM) 자금 이탈", "EEM")
        elif currency == "THB":
            add_macro("달러(DXY) 60일", dxy_chg, True, 3, False, "DXY 강세 (달러 모멘텀)", "DXY")
            add_macro("EEM 60일", eem_chg, True, -10, True, "신흥국(EEM) 자금 이탈", "EEM")
        elif currency == "AUD":
            add_macro("VIX", vix_val, False, 25, False, "VIX 급등 (글로벌 공포)", "VIX")
            add_macro("HYG 60일", hyg_chg, True, -5, True, "하이일드(HYG) 하락", "HYG")
            add_macro("EEM 60일", eem_chg, True, -10, True, "신흥국(EEM) 자금 이탈", "EEM")
            add_macro("금(GOLD) 60일", gold_chg, True, 15, False, "GOLD 급등 (안전자산 수요)", "GOLD")
            add_macro("미국채(TLT) 60일", tlt_chg, True, 10, False, "TLT 급등 (안전선호)", "TLT")
        elif currency == "USD":
            add_macro("EEM 60일", eem_chg, True, -10, True, "신흥국(EEM) 자금 이탈", "EEM")

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
                # 통화별 OOS 검증 결과 (walk-forward, 갭필터 적용 기준)
                #  ※ 개별 통화 OOS CAGR — 합산/look-ahead 착시 배제한 보수적 값
                investor_stats = {
                    'USD': {'win': 85.7, 'cagr': 3.79, 'sharpe': 0.43, 'mdd': 4.72, 'grade': '★★★', 'ok': True},
                    'JPY': {'win': 73.8, 'cagr': 3.05, 'sharpe': 0.17, 'mdd': 4.33, 'grade': '★★',  'ok': True},
                    'AUD': {'win': 66.7, 'cagr': 3.97, 'sharpe': 0.43, 'mdd': 2.81, 'grade': '★★',  'ok': True},
                    'THB': {'win': 58.5, 'cagr': 2.85, 'sharpe': 0.10, 'mdd': 5.33, 'grade': '★',   'ok': True},
                    'EUR': {'win': 0,    'cagr': 0,    'sharpe': 0,    'mdd': 0,    'grade': '—',    'ok': False},
                }
                
                stats = investor_stats.get(currency, {'win': 0, 'cagr': 0, 'sharpe': 0, 'mdd': 0, 'grade': '—', 'ok': False})
                
                stat_msg = f"<div style='padding:12px; background:var(--card-bg); border-radius:8px; border:1px solid var(--border-color);'>"
                stat_msg += f"🌍 <strong>시장 경보 Level {warn_total}</strong><br>"
                if warn_total > 0:
                    stat_msg += f"<span style='color:#E24B4A; font-size:12px;'>" + "<br>".join(active_warnings) + "</span>"
                    stat_msg += f"<hr style='margin:8px 0; border:none; border-top:1px dashed var(--border-color);'>"
                    stat_msg += f"<span style='color:#E24B4A; font-size:12px;'>⚠️ <b>거시 경고 발동 — 시그널 발생해도 신중 진입</b></span>"
                else:
                    stat_msg += f"<span style='color:#1D9E75; font-size:12px;'>🟢 거시 환경 안정</span>"
                stat_msg += f"</div>"
                
                if not stats['ok']:
                    # EUR: 부적합 — 통계 대신 안내
                    stat_msg += f"<br><div style='padding:12px; background:var(--card-bg); border-radius:8px; border:1px solid #E2954B40;'>"
                    stat_msg += f"<span style='font-size:13px; font-weight:700; color:#E2954B'>⚠️ 투자 신호 미제공</span><br>"
                    stat_msg += f"<div style='font-size:12px; margin-top:6px; line-height:1.7; color:var(--text-main)'>"
                    stat_msg += f"유로화(EUR)는 5가지 검증법으로 분석한 결과,<br>"
                    stat_msg += f"평균회귀 전략의 강건한 수익이 확인되지 않았습니다.<br>"
                    stat_msg += f"<span style='color:var(--text-sub); font-size:11px;'>(IS Sharpe 음수 · walk-forward 통과율 43% · 손절 도달률 과다)</span><br>"
                    stat_msg += f"환율 정보와 온도계는 정상 이용 가능합니다."
                    stat_msg += f"</div></div>"
                else:
                    stat_msg += f"<br><div style='padding:12px; background:var(--card-bg); border-radius:8px; border:1px solid var(--border-color);'>"
                    stat_msg += f"<span style='font-size:13px; font-weight:700;'>📊 검증 결과 (20년 백테스트)</span><br>"
                    stat_msg += f"<div style='font-size:12px; margin-top:6px; line-height:1.8;'>"
                    stat_msg += f"• 승률: <strong>{stats['win']:.1f}%</strong><br>"
                    stat_msg += f"• 연수익(CAGR): <strong style='color:#1D9E75'>+{stats['cagr']:.2f}%</strong> <span style='font-size:10px; color:var(--text-sub)'>(무위험수익률 상회)</span><br>"
                    stat_msg += f"• 샤프지수: <strong>{stats['sharpe']:.2f}</strong> · 최대낙폭: <strong>{stats['mdd']:.2f}%</strong>"
                    stat_msg += f"</div></div>"
                
            # ═══════════════════════════════════════
            # 여행자 모드: 온도계 기반 가이드 (v8 — 추세예측 제거, 절감률 강조)
            # ═══════════════════════════════════════
            else:
                saving_val = thermo_data['saving'] if thermo_data else 0.0
                markov_7d = markov_lookup.get(currency, {}).get('7', {}).get(str_total)
                down_prob = None
                if markov_7d:
                    down_prob = markov_7d.get('down', markov_7d.get('cheaper', None))

                # 절감률 표현: '유리/손해' 단어 빼고 수치만 (등급 헤드라인과 충돌 방지)
                sav_word = f"<strong>평균 {saving_val:+.2f}%</strong>"
                # +면 평소보다 싸게, -면 비싸게 샀다는 의미 한 줄
                sav_hint = "<span style='color:var(--text-sub); font-size:12px;'>(+는 평소보다 유리, −는 불리)</span>"

                # ── 등급별 결론 (헤드라인 색만 등급별, 점수 기반 절감률) ──
                if grade in ['A', 'B']:
                    head_color = "#1D9E75"
                    headline = "🟢 지금이 환전하기 아주 좋은 시점이에요" if grade == 'A' else "🟢 지금 환전하기 좋은 편이에요"
                    body = f"이 점수에서 환전했을 때 과거 {sav_word}"
                    action = "💰 출국 전이라면 지금 환전을 추천해요"
                elif grade == 'C':
                    head_color = "#888888"
                    headline = "⚪ 지금은 보통 수준이에요"
                    body = f"이 점수에서는 과거 {sav_word}"
                    action = "🗓️ 일정에 맞춰 환전하면 돼요"
                else:  # D, F
                    head_color = "#E24B4A" if grade == 'F' else "#EF9F27"
                    headline = "🔴 지금은 환전하기 불리해요" if grade == 'F' else "🟡 지금은 다소 불리해요"
                    body = f"이 점수에서 환전했을 때 과거 {sav_word}"
                    if down_prob is not None and isinstance(down_prob, (int, float)) and down_prob >= 55:
                        action = f"⏳ 7일 안에 더 싸질 가능성이 {down_prob}%로, 며칠 기다리는 걸 추천해요"
                    else:
                        action = "⏳ 출국이 급하지 않다면 며칠 기다려 보세요"

                # ── 메시지 조립: 흰색 박스, 헤드라인만 등급색 ──
                stat_msg = f"<div style='padding:14px; background:var(--card-bg); border-radius:10px; border:1px solid var(--border-color);'>"
                stat_msg += f"<div style='font-size:15px; font-weight:800; color:{head_color}; margin-bottom:6px;'>{headline}</div>"
                stat_msg += f"<div style='font-size:13px; color:var(--text-main); line-height:1.6;'>{body}</div>"
                stat_msg += f"<div style='font-size:13px; color:{head_color}; font-weight:700; margin-top:8px;'>{action}</div>"
                stat_msg += f"</div>"

                # ── 시장 경보: 위기(Level 1+)일 때만 ──
                if warn_total > 0:
                    stat_msg += f"<div style='margin-top:10px; padding:12px; background:var(--card-bg); border-radius:8px; border:1px solid #E24B4A40;'>"
                    stat_msg += f"⚠️ <strong style='color:#E24B4A'>시장 경보</strong> — 평소와 다른 위기 신호가 감지됐어요<br>"
                    stat_msg += f"<span style='color:#E24B4A; font-size:12px;'>" + "<br>".join(active_warnings) + "</span><br>"
                    stat_msg += f"<span style='color:var(--text-sub); font-size:11px;'>위기 구간에서는 평소 패턴이 깨질 수 있으니 신중하게 판단하세요.</span>"
                    stat_msg += f"</div>"

        return {
            "price": adj_price, "rsi": int(total), "threshold": int(threshold), "signal": bool(total <= threshold),
            "grade": grade, "is_investable": True, "inv_signal": bool(inv_signal),
            "inv_strategy": inv_strategy, "inv_tpsl": inv_tpsl, "inv_conds": inv_conds, "sub_details": sub_details_dict,
            "macro_items": m_items, "macro_count": warn_total, "ai_message": stat_msg, "stat_details": stat_details,
            "rt_status": rt_status, "rt_valid": rt_valid,
            "grade_cuts": list(GRADE_CUTS[currency]),
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
            _vix_for_signal = macro.get('VIX', {}).get('val', 0)
            inv_signal, strategy, _, _, wr10, tp_price, sl_price, rt_status, rt_valid = calc_inv_signal(df, currency, rt_price, _vix_for_signal)
            
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
        
        strat = INVESTOR_STRATEGIES[currency]
        
        # 차트 하단 지표 = 그 통화의 실제 신호 지표 (RSI 또는 WR + 해당 기간)
        if strat['sig'] == 'RSI':
            df['SIGIND'] = _calc_rsi(df['Close'], strat['period'])
        else:
            df['SIGIND'] = calc_williams_r(df['High'], df['Low'], df['Close'], strat['period'])
        ind_label = f"{strat['sig']}({strat['period']})"
        ind_threshold = strat['th']
        
        # BB (투자자 전략에 맞는 BB)
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
                "rsi": round(float(row['SIGIND']), 1) if not pd.isna(row['SIGIND']) else None
            })
        
        return {
            "currency": currency,
            "bb_period": bb_period,
            "bb_sigma": bb_sigma,
            "ind_label": ind_label,
            "ind_threshold": ind_threshold,
            "data": chart_data
        }
    except Exception as e:
        print(f"Chart Error: {e}")
        return {"currency": currency, "data": []}

@app.get("/api/score_history")
def get_score_history(currency: str = "USD"):
    """과거 시점(어제/1주/1달/1년)의 온도계 점수 비교"""
    try:
        data, macro = get_base_data()
        df = calc_all_indicators(data, currency)
        weights = THERMO_WEIGHTS[currency]
        
        # 과거 시점들 (영업일 기준)
        offsets = {"yesterday": -2, "week": -6, "month": -21, "year": -252}
        summary = {}
        
        for label, offset in offsets.items():
            if len(df) < abs(offset):
                summary[label] = None
                continue
            
            # 해당 시점 기준 가중 합계
            total = 0
            for ind, weight in weights.items():
                if ind == 'SYNC':
                    # SYNC는 시계열 아님 → 중립값 50으로 처리 (보수적)
                    val = 50
                elif ind in df.columns:
                    raw = df[ind].iloc[offset]
                    val = float(raw) if not pd.isna(raw) else 50
                else:
                    val = 50
                total += val * weight
            
            summary[label] = round(total, 1)
        
        return {"summary": summary}
    except Exception as e:
        print(f"History Error: {e}")
        return {"summary": {}}

@app.get("/")
def home():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "index.html")
    return FileResponse(html_path)


@app.on_event("startup")
def startup_event():
    """
    앱 시작 시:
    1. yfinance에서 첫 데이터 즉시 로드 (블로킹)
    2. 백그라운드 워커 스레드 시작 (1분 주기 갱신)
    """
    print("🚀 앱 시작 — yfinance 첫 호출 중... (최대 30초 소요)")
    try:
        data, macro = fetch_from_yfinance()
        with _lock:
            _data_store['data'] = data
            _data_store['macro'] = macro
            _data_store['ready'] = True
            _data_store['last_update'] = time.time()
        print(f"✅ 첫 데이터 로드 완료 ({len(data)}개 통화, {len(macro)}개 매크로)")
    except Exception as e:
        print(f"❌ 첫 로드 실패 (계속 진행 — 다음 사용자 요청 시 재시도): {e}")
    
    # 백그라운드 갱신 스레드 시작 (daemon=True: 앱 종료 시 자동 정리)
    thread = threading.Thread(target=background_updater, daemon=True)
    thread.start()
    print("🔄 백그라운드 갱신 스레드 시작 (1분 주기)")