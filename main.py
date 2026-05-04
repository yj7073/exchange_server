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

def get_base_data():
    current_time = time.time()
    if current_time - cache['time'] < 60 and cache['data']:
        return cache['data'], cache['macro']
    
    tickers = {"USD": "KRW=X", "JPY": "JPYKRW=X", "EUR": "EURKRW=X", "AUD": "AUDKRW=X", "THB": "THBKRW=X"}
    raw_data = {}
    for k, v in tickers.items():
        raw_data[k] = yf.Ticker(v).history(period="2y")['Close'].dropna()
    
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

@app.get("/api/score")
def get_score(currency: str = "USD", d_day: int = 14):
    try:
        data, macro = get_base_data()
        df = pd.DataFrame(data[currency])
        df.columns = ['Close']
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))
        
        curr_price = float(df['Close'].iloc[-1])
        curr_rsi = float(df['RSI'].iloc[-1])
        adj_price = curr_price * (100 if currency == "JPY" else 1)
        
        bb_13_2 = float(df['Close'].rolling(13).mean().iloc[-1] - (df['Close'].rolling(13).std().iloc[-1] * 2))
        bb_12_2 = float(df['Close'].rolling(12).mean().iloc[-1] - (df['Close'].rolling(12).std().iloc[-1] * 2))
        bb_12_15 = float(df['Close'].rolling(12).mean().iloc[-1] - (df['Close'].rolling(12).std().iloc[-1] * 1.5))
        
        vol20_raw = float(df['Close'].pct_change().rolling(20).std().iloc[-1])
        vol60_raw = float(df['Close'].pct_change().rolling(60).std().iloc[-1])
        vol250_raw = float(df['Close'].pct_change().rolling(250).std().iloc[-1])

        inv_signal = False; inv_strategy = ""; inv_tpsl = ""; inv_conds = []

        if currency == "JPY":
            pass_rsi = bool(curr_rsi <= 35)
            pass_bb = bool(curr_price <= bb_13_2)
            inv_signal = pass_rsi and pass_bb
            inv_strategy = "RSI≤35 + BB(13, 2σ) 하향 돌파"
            inv_tpsl = "익절 1.1% / 손절 0.8% / 타임아웃 30일"
            inv_conds = [{"name": "상대강도 (RSI)", "target": "≤ 35.0", "current": f"{curr_rsi:.1f}", "pass": pass_rsi}, {"name": "볼린저 하단 (13, 2σ)", "target": f"≤ {bb_13_2 * 100:.2f}", "current": f"{adj_price:.2f}", "pass": pass_bb}]
        elif currency == "EUR":
            is_low_vol = bool(vol20_raw < vol60_raw)
            if is_low_vol:
                pass_rsi = bool(curr_rsi <= 42)
                pass_bb = bool(curr_price <= bb_12_15)
                inv_signal = pass_rsi and pass_bb
                inv_strategy = "[저변동성] RSI≤42 + BB(12, 1.5σ)"
                inv_tpsl = "익절 0.8% / 손절 0.6%"
                inv_conds = [{"name": "시장 변동성", "target": "저변동성 유지", "current": "저변동성", "pass": True}, {"name": "상대강도 (RSI)", "target": "≤ 42.0", "current": f"{curr_rsi:.1f}", "pass": pass_rsi}, {"name": "볼린저 하단", "target": f"≤ {bb_12_15:.2f}", "current": f"{adj_price:.2f}", "pass": pass_bb}]
            else:
                pass_rsi = bool(curr_rsi <= 38)
                pass_bb = bool(curr_price <= bb_12_2)
                inv_signal = pass_rsi and pass_bb
                inv_strategy = "[고변동성] RSI≤38 + BB(12, 2σ)"
                inv_tpsl = "익절 1.8% / 손절 0.8%"
                inv_conds = [{"name": "시장 변동성", "target": "고변동성 돌파", "current": "고변동성", "pass": True}, {"name": "상대강도 (RSI)", "target": "≤ 38.0", "current": f"{curr_rsi:.1f}", "pass": pass_rsi}, {"name": "볼린저 하단", "target": f"≤ {bb_12_2:.2f}", "current": f"{adj_price:.2f}", "pass": pass_bb}]

        mom5_val = float(df['Close'].pct_change(5).iloc[-1]) * 100
        mom5_score = float(np.clip(((mom5_val + 3) / 6) * 100, 0, 100))
        mom20_val = float(df['Close'].pct_change(20).iloc[-1]) * 100
        mom20_score = float(np.clip(((mom20_val + 12) / 24) * 100, 0, 100))

        sync_pool = ["JPY", "EUR", "THB", "AUD"]
        if currency in sync_pool: sync_pool.remove(currency)
        other_moms = [float(((data[c].iloc[-1] / data[c].iloc[-6]) - 1) * 100) for c in sync_pool if c in data]
        sync_score = float(np.clip(((float(np.mean(other_moms)) + 2) / 4) * 100, 0, 100)) if other_moms else 50
        vol_ratio = vol20_raw / vol250_raw if vol250_raw != 0 else 1
        vol_score = float(np.clip(100 - ((vol_ratio - 0.5) / 1.5) * 100, 0, 100))

        dxy_val = macro.get('DXY', {}).get('val', 100)
        dxy_ma20 = macro.get('DXY', {}).get('ma20', 100)
        dxy_score = float(np.clip(((dxy_val - dxy_ma20) / dxy_ma20 * 100 * 20 + 50), 0, 100))

        s = {"RSI": curr_rsi, "MOM5": mom5_score, "MOM20": mom20_score, "VOL": vol_score, "SYNC": sync_score, "DXY": dxy_score}

        if currency == "JPY": total = (s["RSI"]*0.50) + (s["MOM5"]*0.25) + (s["SYNC"]*0.15) + (s["MOM20"]*0.10)
        elif currency == "EUR": total = (s["RSI"]*0.50) + (s["MOM5"]*0.25) + (s["VOL"]*0.10) + (s["SYNC"]*0.10) + (s["DXY"]*0.05)
        elif currency == "THB": total = (s["RSI"]*0.40) + (s["MOM5"]*0.25) + (s["MOM20"]*0.15) + (s["SYNC"]*0.15) + (s["DXY"]*0.05)
        elif currency == "AUD": total = (s["RSI"]*0.45) + (s["MOM5"]*0.25) + (s["SYNC"]*0.15) + (s["DXY"]*0.10) + (s["MOM20"]*0.05)
        else: total = (s["RSI"]*0.45) + (s["MOM5"]*0.25) + (s["SYNC"]*0.15) + (s["MOM20"]*0.10) + (s["VOL"]*0.05)

        threshold = 50
        if HAS_STATS:
            formula = dday_formulas.get(currency, {})
            for d in sorted([int(k) for k in formula.keys()], reverse=True):
                if d_day >= d:
                    threshold = formula[str(d)]
                    break

        if total <= 25: grade = "A"
        elif total <= 45: grade = "B"
        elif total <= 55: grade = "C"
        elif total <= 75: grade = "D"
        else: grade = "E"

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
            add_macro("HYG 60일", hyg_chg, True, -5, True, "하이일드(HYG) 하락")
            add_macro("EEM 60일", eem_chg, True, -10, True, "신흥국(EEM) 자금 이탈")

        warn_total = sum(w_cnt)
        capped_w_cnt = min(warn_total, 5) 

        investor_guides = {
            0: {"fut": "+0.96%", "holder": "보유 유지", "buyer": "정상 매수 OK"},
            1: {"fut": "+0.29%", "holder": "보유 유지", "buyer": "정상 매수 OK"},
            2: {"fut": "+0.59%", "holder": "보유 유지", "buyer": "정상 매수 OK"},
            3: {"fut": "-1.10%", "holder": "비헤지 유지(유리)", "buyer": "적극 매수 (싸짐!)"},
            4: {"fut": "+0.90%", "holder": "보유 유지", "buyer": "정상 매수 OK"},
            5: {"fut": "+2.15%", "holder": "부분 헤지 고려", "buyer": "소량 매수 시작"}
        }

        sub_details_dict = {}
        for k, v in s.items():
            score_int = int(v)
            sub_details_dict[k] = {"score": score_int}
            if HAS_STATS:
                ind_data = indicator_lookup.get(currency, {}).get(k, {}).get('scores', {}).get(str(score_int))
                if ind_data:
                    sub_details_dict[k].update({"grade": ind_data.get("grade", ""), "win": ind_data.get("win", 0), "saving": ind_data.get("saving", 0.0)})

        stat_msg = ""
        stat_details = {}
        if HAS_STATS:
            str_total = str(int(total))
            thermo_data = indicator_lookup.get(currency, {}).get('THERMO', {}).get('scores', {}).get(str_total)
            if thermo_data:
                stat_details['win_rate'] = thermo_data['win']
                stat_details['saving'] = thermo_data['saving']
                stat_msg = f"과거 통계상 이 구간 진입 시 승률은 <strong>{thermo_data['win']}%</strong> 였습니다."
            
            markov_7d = markov_lookup.get(currency, {}).get('7', {}).get(str_total)
            if markov_7d:
                stat_msg += f"<br>7일 뒤 하락(유리)할 확률: <strong>{markov_7d['cheaper']}%</strong>"
            
            gd = investor_guides[capped_w_cnt]
            stat_msg += f"<br><br><div style='padding:12px; background:var(--card-bg); border-radius:8px; border:1px solid var(--border-color);'>"
            stat_msg += f"🌍 <strong>시장 경보 Level {warn_total}</strong><br>"
            if warn_total > 0:
                stat_msg += f"<span style='color:#E24B4A; font-size:12px;'>" + "<br>".join(active_warnings) + "</span><hr style='margin:8px 0; border:none; border-top:1px dashed var(--border-color);'>"
            else:
                stat_msg += f"<span style='color:#1D9E75; font-size:12px;'>🟢 모든 통화 맞춤형 거시지표가 안정적입니다.</span><hr style='margin:8px 0; border:none; border-top:1px dashed var(--border-color);'>"
            
            stat_msg += f"<span style='font-size:13px; font-weight:700;'>💡 통계 기반 행동 가이드 (60일 후 전망: {gd['fut']})</span><br>"
            buyer_color = "#1D9E75" if capped_w_cnt == 3 else "var(--text-main)"
            stat_msg += f"<span style='font-size:12px;'>• 보유자: {gd['holder']}<br>• <b>매수자: <span style='color:{buyer_color}'>{gd['buyer']}</span></b></span>"
            stat_msg += "</div>"

        return {
            "price": adj_price, "rsi": int(total), "threshold": int(threshold), "signal": bool(total <= threshold),
            "grade": grade, "is_investable": bool(currency in ["JPY", "EUR"]), "inv_signal": bool(inv_signal),
            "inv_strategy": inv_strategy, "inv_tpsl": inv_tpsl, "inv_conds": inv_conds, "sub_details": sub_details_dict,
            "macro_items": m_items, "macro_count": warn_total, "ai_message": stat_msg, "stat_details": stat_details
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"price": 0, "rsi": 50, "threshold": 50, "signal": False, "grade": "C", "is_investable": False, "inv_signal": False, "inv_strategy": "", "inv_tpsl": "", "inv_conds": [], "sub_details": {}, "macro_items": [], "macro_count": 0, "ai_message": "데이터 오류", "stat_details": {}}

@app.get("/api/chart")
def get_chart(currency: str = "USD", days: int = 90):
    """일별 캔들차트 + 볼린저밴드 + RSI 데이터"""
    try:
        data, macro = get_base_data()
        df = pd.DataFrame(data[currency])
        df.columns = ['Close']
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))

        bb_period = 13 if currency == "JPY" else 12
        bb_sigma = 2.0
        
        if currency == "EUR":
            vol20 = float(df['Close'].pct_change().rolling(20).std().iloc[-1])
            vol60 = float(df['Close'].pct_change().rolling(60).std().iloc[-1])
            if vol20 < vol60:
                bb_sigma = 1.5
        
        ma = df['Close'].rolling(bb_period).mean()
        std = df['Close'].rolling(bb_period).std()
        df['MA'] = ma
        df['BB_Upper'] = ma + bb_sigma * std
        df['BB_Lower'] = ma - bb_sigma * std
        
        df = df.tail(days)
        df.dropna(inplace=True)
        
        chart_data = []
        for date, row in df.iterrows():
            chart_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "close": round(float(row['Close']) * (100 if currency == "JPY" else 1), 2),
                "ma": round(float(row['MA']) * (100 if currency == "JPY" else 1), 2),
                "bb_upper": round(float(row['BB_Upper']) * (100 if currency == "JPY" else 1), 2),
                "bb_lower": round(float(row['BB_Lower']) * (100 if currency == "JPY" else 1), 2),
                "rsi": round(float(row['RSI']), 1) if not pd.isna(row['RSI']) else 50
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