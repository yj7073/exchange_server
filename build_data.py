import yfinance as yf
import pandas as pd
import numpy as np
import json
import warnings
import os

warnings.filterwarnings('ignore')

# 🌟 용진님을 위한 추가 마법: outputs 폴더가 없으면 알아서 짠! 하고 만듭니다.
os.makedirs('./outputs', exist_ok=True)
print("📂 통계 데이터를 저장할 './outputs' 폴더가 준비되었습니다.")

# ══════════════════════════════════════════════════════════════════════
#  앱용 JSON Lookup Table 생성
#
#  1. 온도계 1점 단위 절감률/승률
#  2. 개별 지표 1점 단위 절감률/승률
#  3. 마르코프 전이 확률
#  4. D-Day 완화 공식
#  5. ETF 경고 시스템
#
#  출력: ./outputs/ 에 JSON 파일 자동 저장
# ══════════════════════════════════════════════════════════════════════

START_DATE = '2006-01-01'

print("⌛ 야후 파이낸스에서 과거 18년 치 데이터를 불러오는 중... (1~2분 소요)")

tickers = {
    'JPY': 'JPYKRW=X', 'EUR': 'EURKRW=X', 'THB': 'THBKRW=X',
    'AUD': 'AUDKRW=X', 'USD': 'KRW=X',
}

dxy_data = yf.download('DX-Y.NYB', start=START_DATE, progress=False)['Close']
if isinstance(dxy_data, pd.DataFrame): dxy_data = dxy_data.iloc[:, 0]

def calculate_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    return 100 - (100 / (1 + (ema_up / ema_down)))

currency_close = {}
for name, ticker in tickers.items():
    d = yf.download(ticker, start=START_DATE, progress=False)['Close']
    if isinstance(d, pd.DataFrame): d = d.iloc[:, 0]
    currency_close[name] = d
    print(f"  ✅ {name} 데이터 다운로드 완료: {len(d):,}일")

# 온도계 가중치
thermo_weights = {
    'JPY': {'RSI': 0.50, 'MOM5': 0.25, 'SYNC': 0.15, 'MOM20': 0.10},
    'EUR': {'RSI': 0.50, 'MOM5': 0.25, 'VOL': 0.10, 'SYNC': 0.10, 'DXY': 0.05},
    'THB': {'RSI': 0.40, 'MOM5': 0.25, 'SYNC': 0.15, 'MOM20': 0.15, 'DXY': 0.05},
    'AUD': {'RSI': 0.45, 'MOM5': 0.25, 'SYNC': 0.15, 'DXY': 0.10, 'MOM20': 0.05},
    'USD': {'RSI': 0.45, 'MOM5': 0.25, 'SYNC': 0.15, 'MOM20': 0.10, 'VOL': 0.05},
}

indicator_names_kr = {
    'RSI': 'RSI (상대강도)',
    'MOM5': '5일 모멘텀',
    'MOM20': '20일 모멘텀',
    'SYNC': '다통화 동조성',
    'VOL': '변동성',
    'DXY': '달러 방향',
    'THERMO': '종합 온도계',
}

def build_indicators(name):
    prices = currency_close[name]
    df = pd.DataFrame({'PRICE': prices})
    df['RSI'] = calculate_rsi(df['PRICE'])
    
    mom5 = df['PRICE'].pct_change(5) * 100
    df['MOM5'] = ((mom5 + 3) / 6 * 100).clip(0, 100)
    
    mom20 = df['PRICE'].pct_change(20) * 100
    df['MOM20'] = ((mom20 + 12) / 24 * 100).clip(0, 100)
    
    vol20 = df['PRICE'].pct_change().rolling(20).std() * np.sqrt(252) * 100
    vol250 = df['PRICE'].pct_change().rolling(250).std() * np.sqrt(252) * 100
    df['VOL'] = (100 - ((vol20/vol250 - 0.5) / 1.5 * 100)).clip(0, 100)
    
    other_moms = []
    for on, op in currency_close.items():
        if on != name and on != 'USD':
            other_moms.append(op.pct_change(5).reindex(df.index).ffill() * 100)
    if other_moms:
        df['SYNC'] = ((pd.concat(other_moms, axis=1).mean(axis=1) + 2) / 4 * 100).clip(0, 100)
    
    dxy_aligned = dxy_data.reindex(df.index).ffill()
    dxy_ma20 = dxy_aligned.rolling(20).mean()
    df['DXY'] = ((dxy_aligned - dxy_ma20) / dxy_ma20 * 100 * 20 + 50).clip(0, 100)
    
    df.dropna(inplace=True)
    
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
    print(f"  📊 {name} 지표 계산 완료: {len(all_dfs[name]):,}일")


# ██████████████████████████████████████████████████████████████████████
# ██  1. 온도계 + 개별지표 1점 단위 절감률/승률
# ██████████████████████████████████████████████████████████████████████

print(f"\n{'█'*80}")
print(f"█  1. 온도계 + 개별지표 Lookup Table 생성")
print(f"{'█'*80}")

lead = 30
lookup_data = {}

for name in ['JPY', 'EUR', 'THB', 'AUD', 'USD']:
    df = all_dfs[name]
    prices = df['PRICE'].values
    nn = len(df)
    
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
        
        ind_lookup = {}
        for score in range(100):
            data = score_data[score]
            if len(data) < 20: continue
            
            avg = round(np.mean(data), 3)
            win = round(sum(1 for d in data if d > 0) / len(data) * 100, 1)
            amt = round(avg / 100 * 5000000)
            
            if score < 25: grade = "A"
            elif score < 45: grade = "B"
            elif score < 55: grade = "C"
            elif score < 75: grade = "D"
            else: grade = "E"
            
            ind_lookup[str(score)] = {
                "saving": avg,
                "win": win,
                "count": len(data),
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
    print(f"  ✅ {name}: {len(indicators)}개 지표 완료")

# 🌟 경로 수정 적용됨 (./outputs/)
with open('./outputs/indicator_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(lookup_data, f, ensure_ascii=False, indent=2)
print(f"\n  📁 indicator_lookup.json 저장 완료")


# ██████████████████████████████████████████████████████████████████████
# ██  2. 마르코프 전이 확률
# ██████████████████████████████████████████████████████████████████████

print(f"\n{'█'*80}")
print(f"█  2. 마르코프 전이 확률 Lookup Table 생성")
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
            
            a_pct = round((future_scores < 25).sum() / len(future_scores) * 100, 1)
            b_pct = round(((future_scores >= 25) & (future_scores < 45)).sum() / len(future_scores) * 100, 1)
            c_pct = round(((future_scores >= 45) & (future_scores < 55)).sum() / len(future_scores) * 100, 1)
            d_pct = round(((future_scores >= 55) & (future_scores < 75)).sum() / len(future_scores) * 100, 1)
            e_pct = round((future_scores >= 75).sum() / len(future_scores) * 100, 1)
            
            cheaper = round((future_scores < score).sum() / len(future_scores) * 100, 1)
            expensive = round((future_scores > score).sum() / len(future_scores) * 100, 1)
            
            price_chg = round(np.mean((prices[idx + horizon] - prices[idx]) / prices[idx] * 100), 3)
            
            horizon_data[str(score)] = {
                "A": a_pct, "B": b_pct, "C": c_pct, "D": d_pct, "E": e_pct,
                "cheaper": cheaper, "expensive": expensive,
                "price_chg": price_chg,
                "count": len(idx),
            }
        
        currency_markov[str(horizon)] = horizon_data
    
    markov_data[name] = currency_markov
    print(f"  ✅ {name}: 7/14/30일 완료")

# 🌟 경로 수정 적용됨 (./outputs/)
with open('./outputs/markov_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(markov_data, f, ensure_ascii=False, indent=2)
print(f"\n  📁 markov_lookup.json 저장 완료")


# ██████████████████████████████████████████████████████████████████████
# ██  3. D-Day 완화 공식
# ██████████████████████████████████████████████████████████████████████

print(f"\n{'█'*80}")
print(f"█  3. D-Day 완화 공식 JSON 생성")
print(f"{'█'*80}")

dday_formulas = {
    "JPY": { "30": 35, "25": 35, "20": 35, "15": 35, "14": 38, "10": 38, "8": 38, "7": 40, "5": 40, "3": 40, "1": 100 },
    "EUR": { "30": 40, "25": 40, "20": 40, "15": 40, "10": 50, "7": 50, "5": 50, "3": 50, "1": 100 },
    "THB": { "30": 40, "25": 40, "20": 45, "15": 45, "14": 45, "10": 45, "7": 45, "5": 50, "3": 50, "1": 100 },
    "AUD": { "30": 40, "25": 45, "20": 45, "14": 45, "10": 45, "7": 45, "5": 50, "3": 50, "1": 100 },
    "USD": { "30": 50, "25": 50, "20": 50, "15": 50, "14": 50, "10": 50, "7": 50, "5": 50, "3": 50, "1": 100 },
}

# 🌟 경로 수정 적용됨 (./outputs/)
with open('./outputs/dday_formulas.json', 'w', encoding='utf-8') as f:
    json.dump(dday_formulas, f, ensure_ascii=False, indent=2)
print(f"  📁 dday_formulas.json 저장 완료")


# ██████████████████████████████████████████████████████████████████████
# ██  4. ETF 경고 시스템 설정
# ██████████████████████████████████████████████████████████████████████

print(f"\n{'█'*80}")
print(f"█  4. ETF 경고 시스템 JSON 생성")
print(f"{'█'*80}")

etf_config = {
    "warning_indicators": {
        "VIX": {"condition": ">", "threshold": 25, "name_kr": "공포지수"},
        "DXY_CHG60": {"condition": ">", "threshold": 3, "name_kr": "달러강세(60일)"},
        "SP500_CHG60": {"condition": "<", "threshold": -10, "name_kr": "미국증시폭락"},
        "KOSPI_CHG60": {"condition": "<", "threshold": -10, "name_kr": "한국증시폭락"},
        "EEM_CHG60": {"condition": "<", "threshold": -10, "name_kr": "신흥국자금이탈"},
        "OIL_CHG60": {"condition": "<", "threshold": -20, "name_kr": "유가폭락"},
        "GOLD_CHG60": {"condition": ">", "threshold": 10, "name_kr": "금급등"},
    },
    "warning_messages": {
        "0": {"level": "safe", "emoji": "🟢", "msg": "안정 — 정상 투자", "holder": "보유 유지", "buyer": "정상 매수"},
        "1": {"level": "watch", "emoji": "🔵", "msg": "관심 — 모니터링", "holder": "보유 유지", "buyer": "정상 매수"},
        "2": {"level": "caution", "emoji": "🟡", "msg": "주의 — 부분 헤지 고려", "holder": "보유 유지", "buyer": "정상 매수"},
        "3": {"level": "alert", "emoji": "🟠", "msg": "경계 — 60일후 -1.1%, 하락확률 64%", "holder": "비헤지 유리", "buyer": "적극 매수"},
        "4": {"level": "danger", "emoji": "🔴", "msg": "위험 — 변동성 극대", "holder": "보유 유지", "buyer": "정상 매수"},
        "5": {"level": "extreme", "emoji": "🔴🔴", "msg": "극위험 — 위기 진행중", "holder": "부분 헤지", "buyer": "소량 매수"},
    },
    "exit_signals": {
        "warning_drop_from_4": "경고 ≥4개→<4개: 환헤지 축소 시작",
        "warning_duration_30plus": "경고 30일+ 지속 후 해제: 달러 하락 전환",
    },
    "buy_signals": {
        "dxy_down_3pct": {"condition": "DXY_CHG60 < -3", "msg": "달러 약세 추세 — 비헤지 ETF 유리"},
    },
    "historical_stats": {
        "0": {"fut_60": 0.96, "down_prob": 40.7},
        "1": {"fut_60": 0.29, "down_prob": 49.1},
        "2": {"fut_60": 0.59, "down_prob": 58.7},
        "3": {"fut_60": -1.10, "down_prob": 63.8},
        "4": {"fut_60": 0.90, "down_prob": 48.6},
        "5": {"fut_60": 2.15, "down_prob": 44.1},
    },
}

# 🌟 경로 수정 적용됨 (./outputs/)
with open('./outputs/etf_config.json', 'w', encoding='utf-8') as f:
    json.dump(etf_config, f, ensure_ascii=False, indent=2)
print(f"  📁 etf_config.json 저장 완료")


# ██████████████████████████████████████████████████████████████████████
# ██  5. 온도계 가중치 + 등급 체계
# ██████████████████████████████████████████████████████████████████████

print(f"\n{'█'*80}")
print(f"█  5. 온도계 설정 JSON 생성")
print(f"{'█'*80}")

thermo_config = {
    "weights": thermo_weights,
    "grades": {
        "A": {"min": 0, "max": 25, "label": "싸다", "msg": "환전하세요!"},
        "B": {"min": 25, "max": 45, "label": "싼편", "msg": "환전 고려"},
        "C": {"min": 45, "max": 55, "label": "보통", "msg": "급하면 OK"},
        "D": {"min": 55, "max": 75, "label": "비싼편", "msg": "기다려보세요"},
        "E": {"min": 75, "max": 100, "label": "비싸다", "msg": "지금은 비추"},
    },
    "indicator_names": indicator_names_kr,
    "investor_signals": {
        "JPY": {"rsi": 35, "bb_period": 13, "bb_sigma": 2.0, "tp": 0.011, "sl": 0.008},
        "EUR": {
            "normal": {"rsi": 38, "bb_period": 12, "bb_sigma": 2.0, "tp": 0.018, "sl": 0.008},
            "low_vol": {"rsi": 42, "bb_period": 12, "bb_sigma": 1.5, "tp": 0.008, "sl": 0.006},
            "low_vol_condition": "VOL20 < VOL60",
        },
    },
    "optimal_thresholds": {
        "JPY": 35, "EUR": 41, "THB": 46, "AUD": 36, "USD": 50,
    },
}

# 🌟 경로 수정 적용됨 (./outputs/)
with open('./outputs/thermo_config.json', 'w', encoding='utf-8') as f:
    json.dump(thermo_config, f, ensure_ascii=False, indent=2)
print(f"  📁 thermo_config.json 저장 완료")

print(f"\n🎯 5개 JSON 파일 생성 완료!")
print(f"이제 터미널에서 'uvicorn main:app --reload' 또는 'streamlit run app.py'를 실행하세요!")