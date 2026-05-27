import yfinance as yf
import pandas as pd
import numpy as np
import json
import warnings
import os

warnings.filterwarnings('ignore')

os.makedirs('./outputs', exist_ok=True)
print("📂 './outputs' 폴더 준비 완료")

# ══════════════════════════════════════════════════════════════════════
#  build_data.py v7 — WR 기반 온도계
#
#  1. 온도계 + 개별지표 1점 단위 절감률/승률
#  2. 마르코프 전이 확률 (7/14/30일)
#  3. D-Day 완화 공식 v7
#  4. ETF 경고 시스템
#  5. 온도계 설정 (가중치 + 등급)
#
#  출력: ./outputs/ 에 JSON 5개 자동 저장
# ══════════════════════════════════════════════════════════════════════

START_DATE = '2006-01-01'

print("⌛ 야후 파이낸스에서 데이터를 불러오는 중...")

tickers = {
    'JPY': 'JPYKRW=X', 'EUR': 'EURKRW=X', 'THB': 'THBKRW=X',
    'AUD': 'AUDKRW=X', 'USD': 'KRW=X',
}

# 환율 데이터 (OHLC)
currency_data = {}
for name, ticker in tickers.items():
    raw = yf.download(ticker, start=START_DATE, progress=False)
    close = raw['Close'].iloc[:, 0] if isinstance(raw['Close'], pd.DataFrame) else raw['Close']
    high = raw['High'].iloc[:, 0] if isinstance(raw['High'], pd.DataFrame) else raw['High']
    low = raw['Low'].iloc[:, 0] if isinstance(raw['Low'], pd.DataFrame) else raw['Low']
    currency_data[name] = {'close': close, 'high': high, 'low': low}
    print(f"  ✅ {name}: {len(close):,}일")

# ══════════════════════════════════════════════════════════════════════
#  v7 온도계 설정
# ══════════════════════════════════════════════════════════════════════

thermo_weights = {
    'JPY': {'WR10': 0.45, 'SYNC': 0.15, 'DISP10': 0.20, 'WR7': 0.20},
    'EUR': {'WR10': 0.40, 'WR7': 0.60},
    'THB': {'DISP10': 0.60, 'WR10': 0.40},
    'AUD': {'WR10': 1.00},
    'USD': {'WR7': 0.60, 'RSI14': 0.40},
}

# 통화별 ABCDF 등급 경계 (그리드서치 최적화 결과)
# A≤[0], B≤[1], C≤[2], D≤[3], F>[3]
grade_cuts = {
    'EUR': (15, 36, 60, 90),
    'JPY': (22, 36, 52, 76),
    'THB': (32, 43, 56, 72),
    'AUD': (18, 38, 62, 90),
    'USD': (26, 40, 56, 76),
}

def grade_of(score, cuts):
    a, b, c, d = cuts
    if score <= a: return "A"
    elif score <= b: return "B"
    elif score <= c: return "C"
    elif score <= d: return "D"
    else: return "F"

indicator_names_kr = {
    'WR10': 'WR (10일)',
    'WR7': 'WR (7일)',
    'WR14': 'WR (14일)',
    'WR20': 'WR (20일)',
    'MOM3': '3일 모멘텀',
    'MOM5': '5일 모멘텀',
    'SYNC': '다통화 동조성',
    'DISP10': '10일 이격도',
    'DISP20': '20일 이격도',
    'STOCH': '스토캐스틱',
    'RSI7': 'RSI (7일)',
    'RSI14': 'RSI (14일)',
    'THERMO': '종합 온도계',
}

def calculate_williams_r(high, low, close, period):
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return ((highest - close) / (highest - lowest)) * -100

def build_indicators(name):
    """v7 지표 계산 — 통화별 필요한 지표만"""
    d = currency_data[name]
    df = pd.DataFrame({'PRICE': d['close'], 'HIGH': d['high'], 'LOW': d['low']})
    
    # WR (모든 통화 공통)
    for p in [7, 10, 14, 20]:
        wr = calculate_williams_r(df['HIGH'], df['LOW'], df['PRICE'], p)
        df[f'WR{p}'] = (wr + 100).clip(0, 100)  # -100→0, 0→100
    
    # MOM3 (THB용)
    df['MOM3'] = ((df['PRICE'].pct_change(3)*100 + 2) / 4 * 100).clip(0, 100)
    
    # MOM5 (참고용)
    df['MOM5'] = ((df['PRICE'].pct_change(5)*100 + 3) / 6 * 100).clip(0, 100)
    
    # SYNC (다통화 동조성)
    other_names = [n for n in ['JPY', 'EUR', 'THB', 'AUD'] if n != name]
    other_moms = []
    for on in other_names:
        if on in currency_data:
            om = currency_data[on]['close'].pct_change(5).reindex(df.index).ffill() * 100
            other_moms.append(om)
    if other_moms:
        df['SYNC'] = ((pd.concat(other_moms, axis=1).mean(axis=1) + 2) / 4 * 100).clip(0, 100)
    else:
        df['SYNC'] = 50
    
    # DISP (이격도)
    for p in [10, 20]:
        ma = df['PRICE'].rolling(p).mean()
        df[f'DISP{p}'] = ((df['PRICE'] / ma - 0.95) / 0.10 * 100).clip(0, 100)
    
    # STOCH (스토캐스틱)
    low14 = df['LOW'].rolling(14).min()
    high14 = df['HIGH'].rolling(14).max()
    df['STOCH'] = ((df['PRICE'] - low14) / (high14 - low14) * 100).clip(0, 100)
    
    # RSI (USD용 — RSI14)
    for p in [7, 14]:
        delta = df['PRICE'].diff()
        gain = delta.where(delta > 0, 0).rolling(p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
        rs = gain / (loss + 1e-10)
        df[f'RSI{p}'] = (100 - (100 / (1 + rs))).clip(0, 100)
    
    df.dropna(inplace=True)
    
    # 온도계 v7 계산
    w = thermo_weights[name]
    thermo = np.zeros(len(df))
    for ind, weight in w.items():
        if ind in df.columns:
            thermo += df[ind].fillna(50).values * weight
    df['THERMO'] = thermo
    
    return df

all_dfs = {}
for name in tickers.keys():
    all_dfs[name] = build_indicators(name)
    print(f"  📊 {name} v7 지표 계산 완료: {len(all_dfs[name]):,}일")


# ██████████████████████████████████████████████████████████████████████
# ██  1. 온도계 + 개별지표 1점 단위 절감률/승률
# ██████████████████████████████████████████████████████████████████████

print(f"\n{'█'*80}")
print(f"█  1. 온도계 + 개별지표 Lookup Table 생성 (v7)")
print(f"{'█'*80}")

lead = 30
lookup_data = {}

for name in ['JPY', 'EUR', 'THB', 'AUD', 'USD']:
    df = all_dfs[name]
    prices = df['PRICE'].values
    nn = len(df)
    
    # 이 통화에서 사용하는 지표 + THERMO
    indicators = list(thermo_weights[name].keys()) + ['THERMO']
    currency_lookup = {}
    
    for ind_name in indicators:
        ind_vals = df[ind_name].values
        ind_int = ind_vals.astype(int).clip(0, 99)
        
        score_data = {s: [] for s in range(100)}
        
        for dep_idx in range(lead + 250, nn):
            start_idx = dep_idx - lead
            dep_price = prices[dep_idx]
            
            for j in range(start_idx, dep_idx + 1):
                score = ind_int[j]
                saving = (dep_price - prices[j]) / dep_price * 100
                score_data[score].append(saving)
        
        # 점수별 평균 절감률/승률 (1점 단위 원본)
        raw = {}
        for score in range(100):
            data = score_data[score]
            if len(data) < 20: continue
            raw[score] = {
                'avg': round(np.mean(data), 3),
                'win': round(sum(1 for d in data if d > 0) / len(data) * 100, 1),
                'count': len(data),
            }

        ind_lookup = {}
        for score in sorted(raw.keys()):
            avg = raw[score]['avg']
            win = raw[score]['win']
            amt = round(avg / 100 * 5000000)

            if ind_name == 'THERMO':
                grade = grade_of(score, grade_cuts[name])
            else:
                # 개별지표는 참고용 — 단순 5등분 (20/40/60/80)
                grade = grade_of(score, (20, 40, 60, 80))
            
            ind_lookup[str(score)] = {
                "saving": avg,
                "win": win,
                "count": raw[score]['count'],
                "saving_5m": amt,
                "grade": grade,
            }
        
        kr_name = indicator_names_kr.get(ind_name, ind_name)
        weight = thermo_weights[name].get(ind_name, 1.0)
        
        currency_lookup[ind_name] = {
            "name_kr": kr_name,
            "weight": weight,
            "scores": ind_lookup,
        }
    
    lookup_data[name] = currency_lookup
    print(f"  ✅ {name}: {len(indicators)}개 지표 — {', '.join(indicators)}")

with open('./outputs/indicator_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(lookup_data, f, ensure_ascii=False, indent=2)
print(f"\n  📁 indicator_lookup.json 저장 완료")


# ██████████████████████████████████████████████████████████████████████
# ██  2. 마르코프 전이 확률
# ██████████████████████████████████████████████████████████████████████

print(f"\n{'█'*80}")
print(f"█  2. 마르코프 전이 확률 Lookup Table 생성 (v7)")
print(f"{'█'*80}")

markov_data = {}

for name in ['JPY', 'EUR', 'THB', 'AUD', 'USD']:
    df = all_dfs[name]
    thermo_int = df['THERMO'].values.astype(int).clip(0, 99)
    prices = df['PRICE'].values
    nn = len(df)
    
    currency_markov = {}
    
    for horizon in [7, 14, 30]:
        horizon_data = {}
        
        for score in range(100):
            mask = thermo_int == score
            idx = np.where(mask)[0]
            idx = idx[idx + horizon < nn]
            
            if len(idx) < 20: continue
            
            future_scores = thermo_int[idx + horizon]
            
            a_pct = round((future_scores <= 25).sum() / len(future_scores) * 100, 1)
            b_pct = round(((future_scores > 25) & (future_scores <= 40)).sum() / len(future_scores) * 100, 1)
            c_pct = round(((future_scores > 40) & (future_scores <= 55)).sum() / len(future_scores) * 100, 1)
            d_pct = round(((future_scores > 55) & (future_scores <= 70)).sum() / len(future_scores) * 100, 1)
            f_pct = round((future_scores > 70).sum() / len(future_scores) * 100, 1)
            
            cheaper = round((future_scores < score).sum() / len(future_scores) * 100, 1)
            expensive = round((future_scores > score).sum() / len(future_scores) * 100, 1)
            
            price_chg = round(np.mean((prices[idx + horizon] - prices[idx]) / prices[idx] * 100), 3)
            down_prob = round((prices[idx + horizon] < prices[idx]).sum() / len(idx) * 100, 1)
            
            horizon_data[str(score)] = {
                "A": a_pct, "B": b_pct, "C": c_pct, "D": d_pct, "F": f_pct,
                "cheaper": cheaper, "expensive": expensive,
                "price_chg": price_chg, "down": down_prob,
                "count": len(idx),
            }
        
        currency_markov[str(horizon)] = horizon_data
    
    markov_data[name] = currency_markov
    print(f"  ✅ {name}: 7/14/30일 완료")

with open('./outputs/markov_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(markov_data, f, ensure_ascii=False, indent=2)
print(f"\n  📁 markov_lookup.json 저장 완료")


# ██████████████████████████████████████████████████████████████████████
# ██  3. D-Day 완화 공식 v7
# ██████████████████████████████████████████████████████████████████████

print(f"\n{'█'*80}")
print(f"█  3. D-Day 완화 공식 v7")
print(f"{'█'*80}")

# 곡선 v8: D-30~14는 B등급(유리)까지 신호, D-10부터 C(보통)까지 완화, D-1 무조건
# 검증: verify_dday_curve.py — 신호빈도 81~100%, 전 D-day 절감 +, In/Out 양수 확인
dday_formulas = {
    "EUR": {"30": 36, "25": 36, "20": 36, "15": 36, "14": 36, "10": 48, "7": 58, "5": 60, "3": 60, "1": 100},
    "JPY": {"30": 36, "25": 36, "20": 36, "15": 36, "14": 36, "10": 44, "7": 50, "5": 52, "3": 52, "1": 100},
    "AUD": {"30": 38, "25": 38, "20": 38, "15": 38, "14": 38, "10": 50, "7": 60, "5": 62, "3": 62, "1": 100},
    "THB": {"30": 43, "25": 43, "20": 43, "15": 43, "14": 43, "10": 49, "7": 54, "5": 56, "3": 56, "1": 100},
    "USD": {"30": 40, "25": 40, "20": 40, "15": 40, "14": 40, "10": 48, "7": 54, "5": 56, "3": 56, "1": 100},
}

with open('./outputs/dday_formulas.json', 'w', encoding='utf-8') as f:
    json.dump(dday_formulas, f, ensure_ascii=False, indent=2)
print(f"  📁 dday_formulas.json 저장 완료")


# ██████████████████████████████████████████████████████████████████████
# ██  4. ETF 경고 시스템 (통화별 맞춤)
# ██████████████████████████████████████████████████████████████████████

print(f"\n{'█'*80}")
print(f"█  4. ETF 경고 시스템 v7 (통화별 맞춤)")
print(f"{'█'*80}")

etf_config = {
    "warning_indicators": {
        "JPY": [
            {"id": "KOSPI", "name": "KOSPI 60일", "condition": "<", "threshold": -10, "msg": "KOSPI 하락"},
            {"id": "SP500", "name": "S&P500 60일", "condition": "<", "threshold": -10, "msg": "S&P500 하락"},
            {"id": "EEM", "name": "EEM 60일", "condition": "<", "threshold": -10, "msg": "신흥국 자금이탈"},
            {"id": "DXY", "name": "DXY 60일", "condition": ">", "threshold": 5, "msg": "달러 강세"},
            {"id": "HYG", "name": "HYG 60일", "condition": "<", "threshold": -5, "msg": "하이일드 하락"},
            {"id": "NIKKEI", "name": "닛케이 60일", "condition": "<", "threshold": -10, "msg": "닛케이 하락"},
        ],
        "EUR": [
            {"id": "KOSPI", "name": "KOSPI 60일", "condition": "<", "threshold": -10, "msg": "KOSPI 하락"},
            {"id": "SP500", "name": "S&P500 60일", "condition": "<", "threshold": -10, "msg": "S&P500 하락"},
            {"id": "STOXX", "name": "STOXX50 60일", "condition": "<", "threshold": -10, "msg": "유로STOXX 하락"},
            {"id": "EEM", "name": "EEM 60일", "condition": "<", "threshold": -10, "msg": "신흥국 자금이탈"},
        ],
        "THB": [
            {"id": "KOSPI", "name": "KOSPI 60일", "condition": "<", "threshold": -10, "msg": "KOSPI 하락"},
            {"id": "SP500", "name": "S&P500 60일", "condition": "<", "threshold": -10, "msg": "S&P500 하락"},
            {"id": "DXY", "name": "DXY 60일", "condition": ">", "threshold": 3, "msg": "달러 강세"},
            {"id": "EEM", "name": "EEM 60일", "condition": "<", "threshold": -10, "msg": "신흥국 자금이탈"},
        ],
        "AUD": [
            {"id": "KOSPI", "name": "KOSPI 60일", "condition": "<", "threshold": -10, "msg": "KOSPI 하락"},
            {"id": "SP500", "name": "S&P500 60일", "condition": "<", "threshold": -10, "msg": "S&P500 하락"},
            {"id": "VIX_LEVEL", "name": "VIX", "condition": ">", "threshold": 25, "msg": "VIX 급등"},
            {"id": "HYG", "name": "HYG 60일", "condition": "<", "threshold": -5, "msg": "하이일드 하락"},
            {"id": "EEM", "name": "EEM 60일", "condition": "<", "threshold": -10, "msg": "신흥국 자금이탈"},
            {"id": "GOLD", "name": "GOLD 60일", "condition": ">", "threshold": 15, "msg": "금 급등"},
            {"id": "TLT", "name": "TLT 60일", "condition": ">", "threshold": 10, "msg": "미국채 급등"},
        ],
        "USD": [
            {"id": "KOSPI", "name": "KOSPI 60일", "condition": "<", "threshold": -10, "msg": "KOSPI 하락"},
            {"id": "SP500", "name": "S&P500 60일", "condition": "<", "threshold": -10, "msg": "S&P500 하락"},
            {"id": "EEM", "name": "EEM 60일", "condition": "<", "threshold": -10, "msg": "신흥국 자금이탈"},
        ],
    },
    "investor_guides": {
        "JPY": {
            "0": {"fut": "+0.75%", "down": "35.9%", "holder": "보유 유지", "buyer": "정상 매수 OK"},
            "1": {"fut": "+1.50%", "down": "31.3%", "holder": "보유 유지", "buyer": "매수 OK"},
            "2": {"fut": "+1.39%", "down": "27.8%", "holder": "보유 유지", "buyer": "매수 OK"},
            "3": {"fut": "+4.13%", "down": "31.7%", "holder": "비헤지 유지", "buyer": "환헤지 매수"},
            "4": {"fut": "+3.42%", "down": "51.9%", "holder": "부분 헤지", "buyer": "소량 매수"},
            "5": {"fut": "+6.66%", "down": "33.8%", "holder": "부분 헤지", "buyer": "소량 매수"},
        },
        "EUR": {
            "0": {"fut": "+2.21%", "down": "37.4%", "holder": "보유 유지", "buyer": "정상 매수 OK"},
            "1": {"fut": "+3.36%", "down": "45.3%", "holder": "보유 유지", "buyer": "매수 OK"},
            "2": {"fut": "+5.06%", "down": "35.4%", "holder": "보유 유지", "buyer": "매수 OK"},
            "3": {"fut": "-0.48%", "down": "54.8%", "holder": "환헤지 전환", "buyer": "매수 보류"},
            "4": {"fut": "+2.05%", "down": "59.6%", "holder": "환헤지 유지", "buyer": "매수 보류"},
        },
        "THB": {
            "0": {"fut": "+2.18%", "down": "44.3%", "holder": "보유 유지", "buyer": "정상 매수 OK"},
            "1": {"fut": "+4.08%", "down": "41.1%", "holder": "보유 유지", "buyer": "매수 OK"},
            "2": {"fut": "+8.34%", "down": "29.2%", "holder": "보유 유지", "buyer": "매수 OK"},
            "3": {"fut": "+4.75%", "down": "43.6%", "holder": "비헤지 유지", "buyer": "환헤지 매수"},
            "4": {"fut": "+1.67%", "down": "42.6%", "holder": "부분 헤지", "buyer": "소량 매수"},
        },
        "AUD": {
            "0": {"fut": "-0.60%", "down": "45.4%", "holder": "보유 유지", "buyer": "보류 — 경고 시 매수"},
            "1": {"fut": "+8.63%", "down": "24.1%", "holder": "비헤지 유지", "buyer": "적극 매수!"},
            "2": {"fut": "+7.54%", "down": "20.7%", "holder": "비헤지 유지", "buyer": "적극 매수!"},
            "3": {"fut": "+12.26%", "down": "23.4%", "holder": "비헤지 유지", "buyer": "적극 매수!!"},
            "4": {"fut": "+8.24%", "down": "28.8%", "holder": "비헤지 유지", "buyer": "매수 OK"},
            "5": {"fut": "+2.58%", "down": "47.5%", "holder": "부분 헤지", "buyer": "소량 매수"},
        },
        "USD": {
            "0": {"fut": "+3.70%", "down": "26.3%", "holder": "보유 유지", "buyer": "정상 매수 OK"},
            "1": {"fut": "+4.56%", "down": "24.8%", "holder": "보유 유지", "buyer": "매수 OK"},
            "2": {"fut": "+7.56%", "down": "8.6%", "holder": "비헤지 유지", "buyer": "매수 OK"},
            "3": {"fut": "+4.51%", "down": "39.4%", "holder": "비헤지 유지", "buyer": "환헤지 매수"},
        },
    },
    "etf_mapping": {
        "USD": {"unhedge": "TIGER 미국S&P500", "hedge": "TIGER 미국S&P500(H)"},
        "JPY": {"unhedge": "TIGER 일본니케이225", "hedge": "비추 (구조적 손해)"},
        "EUR": {"unhedge": "TIGER 유로스탁50", "hedge": "TIGER 유로스탁50(H)"},
        "AUD": {"unhedge": "KODEX 호주200", "hedge": "해당없음"},
    },
    "hedge_recommendation": {
        "USD": {"default": "비헤지", "reason": "A~F 전 등급 비헤지 유리 (상관계수 0.972)"},
        "JPY": {"default": "비헤지", "reason": "환헤지 ETF 구조적 손해 (-15%)"},
        "EUR": {"default": "환헤지", "reason": "유로 강세 추세, 환헤지가 유리"},
        "AUD": {"default": "비헤지", "reason": "비헤지 전 등급 유리"},
    },
}

with open('./outputs/etf_config.json', 'w', encoding='utf-8') as f:
    json.dump(etf_config, f, ensure_ascii=False, indent=2)
print(f"  📁 etf_config.json 저장 완료")


# ██████████████████████████████████████████████████████████████████████
# ██  5. 온도계 설정 (가중치 + 등급)
# ██████████████████████████████████████████████████████████████████████

print(f"\n{'█'*80}")
print(f"█  5. 온도계 설정 v7")
print(f"{'█'*80}")

thermo_config = {
    "weights": thermo_weights,
    "grade_cuts": {k: list(v) for k, v in grade_cuts.items()},
    "grade_labels": {
        "A": {"label": "매우 유리", "msg": "지금이 환전하기 아주 좋은 시점"},
        "B": {"label": "유리", "msg": "환전하기 좋은 편"},
        "C": {"label": "보통", "msg": "일정에 맞춰 환전"},
        "D": {"label": "불리", "msg": "다소 불리, 여유 있으면 대기"},
        "F": {"label": "매우 불리", "msg": "불리, 며칠 기다려 보기"},
    },
    "indicator_names": indicator_names_kr,
    "optimal_thresholds": {
        "EUR": 15, "JPY": 22, "THB": 32, "AUD": 18, "USD": 26,
    },
    "version": "v8",
    "base_indicator": "Williams %R + RSI(USD)",
    "description": "통화별 맞춤 온도계 v8. USD=WR7/RSI14, THB=DISP10/WR10으로 재최적화. 등급경계 그리드서치 최적화(분위수→손익전환 기반). 추세예측 제거.",
}

with open('./outputs/thermo_config.json', 'w', encoding='utf-8') as f:
    json.dump(thermo_config, f, ensure_ascii=False, indent=2)
print(f"  📁 thermo_config.json 저장 완료")


# ██████████████████████████████████████████████████████████████████████
# ██  완료
# ██████████████████████████████████████████████████████████████████████

print(f"\n{'█'*80}")
print(f"█  완료!")
print(f"{'█'*80}")

print(f"""
  📁 생성된 JSON 파일 (./outputs/):
  
  1. indicator_lookup.json  — 지표별 1점 단위 절감률/승률
     JPY: WR10, SYNC, DISP10, WR7, THERMO
     EUR: WR10, WR7, THERMO
     THB: WR10, MOM3, WR14, STOCH, THERMO
     AUD: WR10, THERMO
     USD: WR10, WR7, DISP20, WR20, THERMO
  
  2. markov_lookup.json     — 7/14/30일 전이확률 + 하락확률
  
  3. dday_formulas.json     — D-Day 완화 경로 v7
     패턴: 5일마다 +5점 완화, D-1 무조건
  
  4. etf_config.json        — 통화별 경고 지표 + ETF 추천
     투자 가이드: 통화별 분리 (down 하락확률 포함)
     환헤지 추천: USD/JPY/AUD 비헤지, EUR 환헤지
  
  5. thermo_config.json     — 온도계 가중치 + 등급 체계

  🚀 이제 'uvicorn main:app --reload'로 서버를 시작하세요!
""")