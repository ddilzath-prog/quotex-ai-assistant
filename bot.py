import os
import re
import io
import json
import time
import base64
import sqlite3
import asyncio
import threading
from datetime import datetime, timezone
from collections import defaultdict

from flask import Flask, jsonify
from PIL import Image

from openai import OpenAI

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "80"))
MIN_SETUP_SCORE = int(os.getenv("MIN_SETUP_SCORE", "70"))

MAX_IMAGE_MB = 8
MAX_IMAGES = 3
COOLDOWN_SECONDS = 5

PORT = int(os.getenv("PORT", "10000"))

DB_PATH = os.getenv("DB_PATH", "signals.db")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")


client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# FLASK HEALTH SERVER
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({
        "bot": "Binance Futures AI Assistant",
        "mode": "Screenshot Analysis",
        "version": "3.0 Unified",
        "status": "online"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


def run_flask():
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False
    )


# ============================================================
# DATABASE
# ============================================================

db_lock = threading.Lock()


def get_db():
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    with db_lock:

        conn = get_db()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,

                symbol TEXT,
                timeframe TEXT,

                signal TEXT,

                long_probability INTEGER,
                short_probability INTEGER,
                confidence INTEGER,

                setup_score INTEGER,

                risk TEXT,

                entry_zone TEXT,
                stop_loss TEXT,
                take_profit TEXT,

                risk_reward REAL,

                reason TEXT,

                outcome TEXT
            )
        """)

        conn.commit()
        conn.close()


# ============================================================
# RUNTIME STATE
# ============================================================

pending_multi = defaultdict(dict)
multi_mode = defaultdict(bool)

last_analysis_time = defaultdict(float)
last_signal_id = defaultdict(int)


# ============================================================
# IMAGE PROCESSING
# ============================================================

def prepare_image(image_bytes: bytes) -> bytes:

    image = Image.open(io.BytesIO(image_bytes))

    image = image.convert("RGB")

    # Keep image large enough for chart analysis
    image.thumbnail((2400, 2400))

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=90,
        optimize=True
    )

    result = output.getvalue()

    # Safety reduction if still too large
    if len(result) > MAX_IMAGE_MB * 1024 * 1024:

        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=75,
            optimize=True
        )

        result = output.getvalue()

    return result


def image_to_base64(image_bytes: bytes) -> str:

    return base64.b64encode(image_bytes).decode("utf-8")


# ============================================================
# TIMEFRAME DETECTION
# ============================================================

def extract_timeframe(text):

    if not text:
        return None

    text = text.lower()

    patterns = [
        r"\b1m\b",
        r"\b3m\b",
        r"\b5m\b",
        r"\b15m\b",
        r"\b30m\b",
        r"\b1h\b",
        r"\b2h\b",
        r"\b4h\b",
        r"\b6h\b",
        r"\b12h\b",
        r"\b1d\b"
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return None


# ============================================================
# AI SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are an extremely strict Binance Futures screenshot-analysis engine.

Your purpose is NOT to guarantee profit.

Your job is to analyze only what is actually visible in the supplied
chart screenshots and produce a conservative trading setup assessment.

Never invent:

- price
- volume
- order book
- long/short ratio
- indicators
- support/resistance
- liquidity
- candles
- timeframe
- market structure
- news
- fundamentals

If something is not visible or cannot be reliably inferred from the chart,
return "Unclear" or "Unavailable".

============================================================
CORE ANALYSIS
============================================================

Analyze:

1. Market trend
2. Market structure
3. Higher-high / higher-low
4. Lower-high / lower-low
5. Break of structure
6. Change of character
7. Support
8. Resistance
9. Supply / demand zones when visually justified
10. Momentum
11. Candle psychology
12. Wick rejection
13. Body strength
14. Consecutive candle pressure
15. Absorption
16. Breakout
17. Fake breakout
18. Liquidity sweep
19. Stop-hunt style movement
20. Trap
21. Exhaustion
22. Volume only if visible
23. Order book only if visible
24. Long/short ratio only if visible

============================================================
CANDLE PSYCHOLOGY
============================================================

Study:

- strong bullish bodies
- strong bearish bodies
- small bodies
- doji
- long upper wick
- long lower wick
- rejection candles
- engulfing behavior
- consecutive candles
- failed continuation
- candle close location

Interpret the candles as evidence of:

- buying pressure
- selling pressure
- hesitation
- rejection
- absorption
- possible exhaustion

Do NOT treat one candle as enough evidence by itself.

============================================================
MARKET / TRADER PSYCHOLOGY
============================================================

Analyze visible evidence of:

- FOMO buying
- panic selling
- late entries
- trapped buyers
- trapped sellers
- stop-loss hunting
- liquidity sweep
- breakout chasing
- rejection after aggressive move
- crowd positioning
- absorption
- fear-driven selling
- greed-driven buying

These are behavioral interpretations of chart structure,
NOT guaranteed facts about individual traders.

Only report them when chart evidence supports them.

============================================================
FINANCIAL / MARKET PSYCHOLOGY
============================================================

Consider:

- momentum chasing
- risk-off behavior visible in price action
- aggressive expansion
- compression before expansion
- distribution
- accumulation
- capitulation-style movement
- failed breakout
- failed breakdown
- liquidity grab
- continuation pressure
- exhaustion

Do not invent macroeconomic information.

============================================================
SUPPORT / RESISTANCE
============================================================

Look for:

- repeated reactions
- swing highs
- swing lows
- rejection zones
- consolidation boundaries
- breakout levels
- retest zones

Do not call a level "confirmed" unless chart evidence supports it.

============================================================
BREAKOUT LOGIC
============================================================

Distinguish:

- confirmed breakout
- weak breakout
- fake breakout
- liquidity sweep
- no breakout

A wick through a level followed by rejection can indicate a
possible false breakout / liquidity sweep.

============================================================
TRAP LOGIC
============================================================

Detect:

- bull trap
- bear trap
- possible trap

If a trap is reasonably possible, prefer NO TRADE.

============================================================
EXHAUSTION
============================================================

Check for:

- extended move
- repeated large candles
- shrinking bodies
- long rejection wicks
- failed continuation
- momentum loss
- divergence-like visual behavior if actually visible

High exhaustion should strongly reduce trade confidence.

============================================================
MULTI-TIMEFRAME
============================================================

If multiple screenshots are supplied:

Compare all available timeframes.

Look for:

- trend alignment
- structure alignment
- momentum alignment
- support/resistance alignment
- breakout alignment

If timeframes materially disagree, return:

mtf_alignment = "Mixed"

and prefer NO TRADE.

============================================================
ENTRY / SL / TP
============================================================

Only provide:

- entry zone
- stop loss
- take profit
- risk/reward

when they can be reasonably derived from visible chart structure.

Never invent exact prices.

If unavailable:

"UNKNOWN"

============================================================
SIGNAL RULE
============================================================

Candidate signals:

LONG
SHORT
NO TRADE

A high confidence number alone is NOT enough.

The final system should prefer:

QUALITY > QUANTITY

When evidence conflicts:

NO TRADE

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Schema:

{
  "symbol": "ETHUSDT",
  "market": "BINANCE FUTURES",
  "timeframe": "15m",

  "signal": "LONG | SHORT | NO TRADE",
"long_probability": 0,
"short_probability": 0,
"confidence": 0,
"candidate_direction": "LONG | SHORT | NONE",
"candidate_confidence": 0,
"setup_score": 0,

  "trend": "Bullish",
  "market_structure": "Strong",
  "candle_psychology": "Bullish",
  "momentum": "Strong",

  "support": "Confirmed",
  "support_zone": "text or UNKNOWN",

  "resistance": "Confirmed",
  "resistance_zone": "text or UNKNOWN",

  "breakout": "Confirmed Up",
  "trap": "None",

  "continuation": "Likely",
  "reversal": "Unlikely",

  "exhaustion": "Low",

  "volume": "Strong",
  "order_book": "Bullish",
  "long_short_ratio": "Bullish Bias",

  "mtf_alignment": "Single timeframe",

  "data_quality": "Good",

  "risk": "Low",

  "entry_zone": "text or UNKNOWN",
  "stop_loss": "text or UNKNOWN",
  "take_profit": "text or UNKNOWN",
  "risk_reward": 0,

  "timeframe_views": [
    {
      "timeframe": "15m",
      "bias": "Bullish",
      "confidence": 0
    }
  ],

  "evidence": [
    "short factual observation",
    "short factual observation",
    "short factual observation"
  ],

  "reason": "short factual explanation"
}

Probability values are estimates from visual evidence,
NOT guaranteed win probabilities.

Never return commentary outside JSON.
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str):

    text = text.strip()

    # Remove markdown fences if model accidentally adds them
    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```$",
        "",
        text
    )

    text = text.strip()

    try:
        return json.loads(text)

    except Exception:
        pass

    # Try extracting first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        candidate = text[start:end + 1]

        return json.loads(candidate)

    raise ValueError("AI did not return valid JSON")


# ============================================================
# NORMALIZATION
# ============================================================

def safe_int(value, default=0):

    try:
        return int(float(value))
    except Exception:
        return default


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return None


def normalize_analysis(data):

    if not isinstance(data, dict):
        raise ValueError("Invalid AI response")

    fields = {
        "symbol": "UNKNOWN",
        "market": "BINANCE FUTURES",
        "timeframe": "UNKNOWN",
        "signal": "NO TRADE",

        "trend": "Unclear",
        "market_structure": "Unclear",
        "candle_psychology": "Unclear",
        "momentum": "Unclear",

        "support": "Unclear",
        "support_zone": "UNKNOWN",

        "resistance": "Unclear",
        "resistance_zone": "UNKNOWN",

        "breakout": "Unclear",
        "trap": "Unclear",

        "continuation": "Unclear",
        "reversal": "Unclear",
        "exhaustion": "Unclear",

        "volume": "Unclear",
        "order_book": "Unclear",
        "long_short_ratio": "Unavailable",

        "mtf_alignment": "Insufficient",
        "data_quality": "Poor",

        "risk": "High",

        "entry_zone": "UNKNOWN",
        "stop_loss": "UNKNOWN",
        "take_profit": "UNKNOWN",

        "reason": ""
    }

    for key, default in fields.items():

        if key not in data:
            data[key] = default

    # Normalize probabilities
    long_p = safe_int(data.get("long_probability"), 0)
    short_p = safe_int(data.get("short_probability"), 0)

    long_p = max(0, min(100, long_p))
    short_p = max(0, min(100, short_p))

    total = long_p + short_p

    if total > 0:

        long_p = round((long_p / total) * 100)
        short_p = 100 - long_p

    else:

        long_p = 0
        short_p = 0

    data["long_probability"] = long_p
    data["short_probability"] = short_p

    data["confidence"] = max(
        0,
        min(
            100,
            safe_int(data.get("confidence"), 0)
        )
    )

    signal = str(data.get("signal", "NO TRADE")).upper()

    if signal not in ["LONG", "SHORT", "NO TRADE"]:

        signal = "NO TRADE"

    data["signal"] = signal

    rr = safe_float(data.get("risk_reward"))

    if rr is not None and rr < 0:
        rr = None

    data["risk_reward"] = rr

    if not isinstance(data.get("evidence"), list):
        data["evidence"] = []

    if not isinstance(data.get("timeframe_views"), list):
        data["timeframe_views"] = []

    return data


# ============================================================
# MULTI-TIMEFRAME CHECK
# ============================================================

def calculate_mtf_alignment(data):

    views = data.get("timeframe_views", [])

    biases = []

    for view in views:

        if not isinstance(view, dict):
            continue

        bias = str(view.get("bias", "")).strip()

        if bias in ["Bullish", "Bearish"]:
            biases.append(bias)

    if len(biases) <= 1:

        return data.get(
            "mtf_alignment",
            "Single timeframe"
        )

    if all(x == "Bullish" for x in biases):

        return "Aligned Bullish"

    if all(x == "Bearish" for x in biases):

        return "Aligned Bearish"

    return "Mixed"


# ============================================================
# SETUP SCORE
# ============================================================

def calculate_setup_score(data):

    signal = data["signal"]

    # Candidate direction based on probabilities
    if signal == "NO TRADE":

        if data["long_probability"] > data["short_probability"]:
            direction = "LONG"

        elif data["short_probability"] > data["long_probability"]:
            direction = "SHORT"

        else:
            return 0

    else:

        direction = signal

    score = 0

    trend = data.get("trend")
    structure = data.get("market_structure")
    candle = data.get("candle_psychology")
    momentum = data.get("momentum")

    support = data.get("support")
    resistance = data.get("resistance")

    breakout = data.get("breakout")
    trap = data.get("trap")

    exhaustion = data.get("exhaustion")
    volume = data.get("volume")

    confirmation = data.get("confirmation", "Unclear")

    risk = data.get("risk")

    mtf = calculate_mtf_alignment(data)

    quality = data.get("data_quality")

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    if quality == "Good":
        score += 10

    elif quality == "Fair":
        score += 5

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    if structure == "Strong":
        score += 15

    elif structure == "Moderate":
        score += 10

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if direction == "LONG":

        if trend == "Bullish":
            score += 15

        elif trend == "Range":
            score += 5

    elif direction == "SHORT":

        if trend == "Bearish":
            score += 15

        elif trend == "Range":
            score += 5

    # --------------------------------------------------------
    # CANDLE PSYCHOLOGY
    # --------------------------------------------------------

    if direction == "LONG" and candle == "Bullish":
        score += 15

    elif direction == "SHORT" and candle == "Bearish":
        score += 15

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    if momentum == "Strong":
        score += 15

    elif momentum == "Moderate":
        score += 10

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    if direction == "LONG" and support == "Confirmed":
        score += 10

    if direction == "SHORT" and resistance == "Confirmed":
        score += 10

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    if direction == "LONG" and breakout == "Confirmed Up":
        score += 10

    if direction == "SHORT" and breakout == "Confirmed Down":
        score += 10

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if volume == "Strong":
        score += 5

    elif volume == "Normal":
        score += 3

    # --------------------------------------------------------
    # CONFIRMATION
    # --------------------------------------------------------

    if confirmation == "Strong":
        score += 10

    elif confirmation == "Good":
        score += 7

    # --------------------------------------------------------
    # MULTI TIMEFRAME
    # --------------------------------------------------------

    if direction == "LONG" and mtf == "Aligned Bullish":
        score += 10

    elif direction == "SHORT" and mtf == "Aligned Bearish":
        score += 10

    elif mtf == "Single timeframe":
        score += 5

    # --------------------------------------------------------
    # PENALTIES
    # --------------------------------------------------------

    if trap in [
        "Bull Trap",
        "Bear Trap",
        "Possible Trap"
    ]:
        score -= 25

    if exhaustion == "High":
        score -= 20

    elif exhaustion == "Medium":
        score -= 5

    if risk == "High":
        score -= 20

    if data.get("data_quality") == "Poor":
        score -= 20

    if mtf == "Mixed":
        score -= 25

    score = max(0, min(100, score))

    return score


# ============================================================
# STRICT QUALITY GATE
# ============================================================

def quality_gate(data):

    original_signal = data["signal"]

    # If AI already says NO TRADE
    if original_signal == "NO TRADE":

        data["signal"] = "NO TRADE"

        return data

    confidence = data["confidence"]

    if confidence < MIN_CONFIDENCE:

        data["signal"] = "NO TRADE"
        data["reason"] = (
            f"Confidence below minimum threshold "
            f"({MIN_CONFIDENCE}%)."
        )

        return data

    if data.get("data_quality") == "Poor":

        data["signal"] = "NO TRADE"
        data["reason"] = "Chart data quality is insufficient."

        return data

    if data.get("risk") == "High":

        data["signal"] = "NO TRADE"
        data["reason"] = "Risk is too high."

        return data

    if data.get("market_structure") in [
        "Weak",
        "Unclear"
    ]:

        data["signal"] = "NO TRADE"
        data["reason"] = "Market structure is not sufficiently clear."

        return data

    if data.get("candle_psychology") in [
        "Mixed",
        "Unclear"
    ]:

        data["signal"] = "NO TRADE"
        data["reason"] = "Candle psychology is conflicting or unclear."

        return data

    if data.get("momentum") == "Unclear":

        data["signal"] = "NO TRADE"
        data["reason"] = "Momentum is unclear."

        return data

    if data.get("exhaustion") == "High":

        data["signal"] = "NO TRADE"
        data["reason"] = "High exhaustion detected."

        return data

    if data.get("trap") in [
        "Bull Trap",
        "Bear Trap",
        "Possible Trap"
    ]:

        data["signal"] = "NO TRADE"
        data["reason"] = "Potential trap detected."

        return data

    if data.get("breakout") == "False Breakout":

        data["signal"] = "NO TRADE"
        data["reason"] = "False breakout detected."

        return data

    mtf = calculate_mtf_alignment(data)

    if mtf == "Mixed":

        data["signal"] = "NO TRADE"
        data["reason"] = "Timeframes are conflicting."

        return data

    signal = original_signal

    trend = data.get("trend")
    candle = data.get("candle_psychology")
    reversal = data.get("reversal")

    # --------------------------------------------------------
    # LONG FILTER
    # --------------------------------------------------------

    if signal == "LONG":

        if trend == "Bearish" and reversal != "Likely":

            data["signal"] = "NO TRADE"
            data["reason"] = (
                "LONG conflicts with bearish structure/trend."
            )

            return data

        if candle != "Bullish":

            data["signal"] = "NO TRADE"
            data["reason"] = (
                "LONG lacks bullish candle confirmation."
            )

            return data

        if data.get("momentum") == "Weak":

            data["signal"] = "NO TRADE"
            data["reason"] = (
                "LONG lacks sufficient momentum."
            )

            return data

    # --------------------------------------------------------
    # SHORT FILTER
    # --------------------------------------------------------

    if signal == "SHORT":

        if trend == "Bullish" and reversal != "Likely":

            data["signal"] = "NO TRADE"
            data["reason"] = (
                "SHORT conflicts with bullish structure/trend."
            )

            return data

        if candle != "Bearish":

            data["signal"] = "NO TRADE"
            data["reason"] = (
                "SHORT lacks bearish candle confirmation."
            )

            return data

        if data.get("momentum") == "Weak":

            data["signal"] = "NO TRADE"
            data["reason"] = (
                "SHORT lacks sufficient momentum."
            )

            return data

    # --------------------------------------------------------
    # SETUP SCORE
    # --------------------------------------------------------

    setup_score = calculate_setup_score(data)

    data["setup_score"] = setup_score

    if setup_score < MIN_SETUP_SCORE:

        data["signal"] = "NO TRADE"

        data["reason"] = (
            f"Setup score {setup_score}/100 is below "
            f"the minimum {MIN_SETUP_SCORE}/100."
        )

        return data

    return data


# ============================================================
# OPENAI ANALYSIS
# ============================================================

def analyze_images(images):

    """
    images:
        [
            ("5m", bytes),
            ("15m", bytes),
            ("1h", bytes)
        ]
    """

    prepared = []

    for label, raw in images:

        image = prepare_image(raw)

        prepared.append(
            (
                label,
                image
            )
        )

    labels_text = []

    for index, (label, _) in enumerate(prepared, start=1):

        labels_text.append(
            f"Image {index}: timeframe label = {label}"
        )

    user_prompt = f"""
Analyze the supplied Binance Futures chart screenshot(s).

{chr(10).join(labels_text)}

IMPORTANT:

- Use only visible evidence.
- Do not invent hidden data.
- If volume/order book/long-short ratio is not visible, mark it Unclear/Unavailable.
- Analyze the full visible candle history, not just the newest candle.
- Give special attention to candle psychology, market psychology,
  trader behavior, liquidity sweeps, traps, exhaustion and continuation.
- Compare timeframes when multiple screenshots are supplied.
- Prefer NO TRADE when evidence conflicts.
- Never guarantee profit.

Return ONLY JSON according to the required schema.
"""

    content = [
        {
            "type": "input_text",
            "text": user_prompt
        }
    ]

    for label, image in prepared:

        encoded = image_to_base64(image)

        content.append(
            {
                "type": "input_text",
                "text": f"Chart image timeframe label: {label}"
            }
        )

        content.append(
            {
                "type": "input_image",
                "image_url": (
                    "data:image/jpeg;base64,"
                    + encoded
                ),
                "detail": "high"
            }
        )

    response = client.responses.create(
        model=OPENAI_MODEL,

        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": SYSTEM_PROMPT
                    }
                ]
            },
            {
                "role": "user",
                "content": content
            }
        ]
    )

    raw_text = response.output_text

    data = extract_json(raw_text)

    data = normalize_analysis(data)

    # Determine MTF status locally too
    data["mtf_alignment"] = calculate_mtf_alignment(data)

    # Apply local setup score
    data["setup_score"] = calculate_setup_score(data)

    # Apply strict final filter
    data = quality_gate(data)

    return data


# ============================================================
# DATABASE RECORDING
# ============================================================

def record_signal(chat_id, data):

    with db_lock:

        conn = get_db()

        cursor = conn.execute(
            """
            INSERT INTO signals (
                chat_id,
                created_at,
                symbol,
                timeframe,
                signal,
                long_probability,
                short_probability,
                confidence,
                setup_score,
                risk,
                entry_zone,
                stop_loss,
                take_profit,
                risk_reward,
                reason,
                outcome
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                datetime.now(timezone.utc).isoformat(),

                data.get("symbol"),
                data.get("timeframe"),

                data.get("signal"),

                data.get("long_probability"),
                data.get("short_probability"),

                data.get("confidence"),
                data.get("setup_score"),

                data.get("risk"),

                data.get("entry_zone"),
                data.get("stop_loss"),
                data.get("take_profit"),

                data.get("risk_reward"),

                data.get("reason"),

                None
            )
        )

        signal_id = cursor.lastrowid

        conn.commit()
        conn.close()

    last_signal_id[chat_id] = signal_id

    return signal_id


# ============================================================
# FORMAT TELEGRAM RESULT
# ============================================================

def format_result(data, signal_id):

    signal = data["signal"]

    if signal == "LONG":
        emoji = "🟢"
    elif signal == "SHORT":
        emoji = "🔴"
    else:
        emoji = "⚪"

    rr = data.get("risk_reward")

    if rr is None:
        rr_text = "UNKNOWN"
    else:
        rr_text = f"{rr:.2f}"

    evidence = data.get("evidence", [])

    evidence_lines = []

    for item in evidence[:3]:

        evidence_lines.append(
            f"• {str(item)[:180]}"
        )

    if not evidence_lines:
        evidence_lines.append(
            "• No additional evidence provided."
        )

    text = f"""
{emoji} BINANCE FUTURES AI

━━━━━━━━━━━━━━━━
SIGNAL: {signal}
━━━━━━━━━━━━━━━━

Symbol: {data.get("symbol")}
Timeframe: {data.get("timeframe")}

LONG: {data.get("long_probability")}%
SHORT: {data.get("short_probability")}%

Confidence: {data.get("confidence")}%
Setup Score: {data.get("setup_score")}/100

MTF: {data.get("mtf_alignment")}
Data Quality: {data.get("data_quality")}

━━━━━━━━━━━━━━━━
MARKET ANALYSIS
━━━━━━━━━━━━━━━━

Trend: {data.get("trend")}
Structure: {data.get("market_structure")}
Momentum: {data.get("momentum")}
Candle Psychology: {data.get("candle_psychology")}

Support: {data.get("support")}
Resistance: {data.get("resistance")}

Breakout: {data.get("breakout")}
Trap: {data.get("trap")}
Exhaustion: {data.get("exhaustion")}

Volume: {data.get("volume")}
Order Book: {data.get("order_book")}
Long/Short: {data.get("long_short_ratio")}

Risk: {data.get("risk")}

━━━━━━━━━━━━━━━━
TRADE LEVELS
━━━━━━━━━━━━━━━━

Entry: {data.get("entry_zone")}
SL: {data.get("stop_loss")}
TP: {data.get("take_profit")}
R:R: {rr_text}

━━━━━━━━━━━━━━━━
MARKET PSYCHOLOGY
━━━━━━━━━━━━━━━━

{chr(10).join(evidence_lines)}

━━━━━━━━━━━━━━━━
REASON
━━━━━━━━━━━━━━━━

{data.get("reason")}

━━━━━━━━━━━━━━━━
Signal ID: #{signal_id}

Paper/demo testing recommended.
Confidence is NOT a guaranteed win probability.
"""

    return text.strip()


# ============================================================
# /START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        """
🤖 Binance Futures AI Assistant V3

Send a Binance Futures chart screenshot.

The AI analyzes:

• Candle Psychology
• Market Psychology
• Trader Behavior
• Market Structure
• Support / Resistance
• Momentum
• Volume
• Breakout / Fake Breakout
• Liquidity Sweep
• Trap
• Exhaustion
• Multi-Timeframe Alignment
• Risk / Reward

Commands:

/multi - Multi-timeframe mode
/analyze - Analyze queued charts
/single - Return to single screenshot mode
/history - Signal history
/stats - Performance statistics
/result WIN - Mark latest trade WIN
/result LOSS - Mark latest trade LOSS
/result INVALID - Mark latest trade INVALID
/status - Bot status
/clear - Clear pending screenshots
/help - Help

QUALITY > QUANTITY

If evidence is weak or conflicting:
NO TRADE
""".strip()
    )


# ============================================================
# /HELP
# ============================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await start_command(update, context)


# ============================================================
# /STATUS
# ============================================================

async def status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    chat_id = update.effective_chat.id

    queued = len(pending_multi.get(chat_id, {}))

    await update.message.reply_text(
        f"""
🟢 BOT STATUS

Version: V3 Unified
Mode: Screenshot Analysis
Model: {OPENAI_MODEL}

Minimum Confidence: {MIN_CONFIDENCE}%
Minimum Setup Score: {MIN_SETUP_SCORE}/100

Multi-Timeframe Mode:
{"ON" if multi_mode[chat_id] else "OFF"}

Queued Images: {queued}

Order Execution:
OFF

This bot generates analysis/signals only.
""".strip()
    )


# ============================================================
# /MULTI
# ============================================================

async def multi_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    chat_id = update.effective_chat.id

    pending_multi[chat_id].clear()

    multi_mode[chat_id] = True

    await update.message.reply_text(
        """
📊 MULTI-TIMEFRAME MODE ON

Send up to 3 screenshots.

Recommended:

1️⃣ 5m
2️⃣ 15m
3️⃣ 1h

You can add the timeframe in the caption:

5m
15m
1h

After sending them:

/analyze

The AI will compare the timeframes.

If they conflict → NO TRADE.
""".strip()
    )


# ============================================================
# /SINGLE
# ============================================================

async def single_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    chat_id = update.effective_chat.id

    pending_multi[chat_id].clear()

    multi_mode[chat_id] = False

    await update.message.reply_text(
        "📷 Single screenshot mode ON."
    )


# ============================================================
# /ANALYZE
# ============================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    chat_id = update.effective_chat.id

    items = pending_multi.get(chat_id, {})

    if not items:

        await update.message.reply_text(
            "No queued screenshots. Use /multi first."
        )

        return

    images = list(items.items())

    pending_multi[chat_id].clear()

    multi_mode[chat_id] = False

    await run_analysis(
        update,
        images
    )


# ============================================================
# RUN ANALYSIS
# ============================================================

async def run_analysis(
    update: Update,
    images
):

    chat_id = update.effective_chat.id

    now = time.time()

    if now - last_analysis_time[chat_id] < COOLDOWN_SECONDS:

        await update.message.reply_text(
            "⏳ Please wait a few seconds before another analysis."
        )

        return

    last_analysis_time[chat_id] = now

    if len(images) > MAX_IMAGES:

        images = images[:MAX_IMAGES]

    await update.message.reply_text(
        "🔎 Analyzing chart psychology + market structure..."
    )

    try:

        data = await asyncio.to_thread(
            analyze_images,
            images
        )

        signal_id = record_signal(
            chat_id,
            data
        )

        result = format_result(
            data,
            signal_id
        )

        await update.message.reply_text(
            result
        )

    except Exception as e:

        print("ANALYSIS ERROR:", repr(e))

        await update.message.reply_text(
            """
❌ Analysis failed.

Possible reasons:

• OpenAI API error
• Invalid image
• Temporary server issue
• AI returned invalid JSON

Please try the screenshot again.
""".strip()
        )


# ============================================================
# PHOTO HANDLER
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    chat_id = update.effective_chat.id

    if not update.message.photo:

        return

    photo = update.message.photo[-1]

    telegram_file = await photo.get_file()

    image_bytes = await telegram_file.download_as_bytearray()

    image_bytes = bytes(image_bytes)

    caption = update.message.caption or ""

    timeframe = extract_timeframe(caption)

    # --------------------------------------------------------
    # MULTI MODE
    # --------------------------------------------------------

    if multi_mode[chat_id]:

        if len(pending_multi[chat_id]) >= MAX_IMAGES:

            await update.message.reply_text(
                f"Maximum {MAX_IMAGES} screenshots queued."
            )

            return

        if timeframe is None:

            timeframe = (
                f"image_{len(pending_multi[chat_id]) + 1}"
            )

        pending_multi[chat_id][timeframe] = image_bytes

        await update.message.reply_text(
            f"""
📥 Screenshot saved

Timeframe: {timeframe}
Queued: {len(pending_multi[chat_id])}/{MAX_IMAGES}

Send another screenshot or use:

/analyze
""".strip()
        )

        return

    # --------------------------------------------------------
    # SINGLE MODE
    # --------------------------------------------------------

    label = timeframe or "single"

    await run_analysis(
        update,
        [
            (
                label,
                image_bytes
            )
        ]
    )


# ============================================================
# /RESULT
# ============================================================

async def result_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    chat_id = update.effective_chat.id

    if not context.args:

        await update.message.reply_text(
            "Usage:\n/result WIN\n/result LOSS\n/result INVALID"
        )

        return

    outcome = context.args[0].upper()

    if outcome not in [
        "WIN",
        "LOSS",
        "INVALID"
    ]:

        await update.message.reply_text(
            "Use WIN, LOSS or INVALID."
        )

        return

    with db_lock:

        conn = get_db()

        row = conn.execute(
            """
            SELECT id, signal
            FROM signals
            WHERE chat_id = ?
              AND signal IN ('LONG', 'SHORT')
              AND outcome IS NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (chat_id,)
        ).fetchone()

        if not row:

            conn.close()

            await update.message.reply_text(
                "No unresolved LONG/SHORT signal found."
            )

            return

        conn.execute(
            """
            UPDATE signals
            SET outcome = ?
            WHERE id = ?
            """,
            (
                outcome,
                row["id"]
            )
        )

        conn.commit()
        conn.close()

    await update.message.reply_text(
        f"✅ Signal #{row['id']} marked as {outcome}."
    )


# ============================================================
# /STATS
# ============================================================

async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    chat_id = update.effective_chat.id

    with db_lock:

        conn = get_db()

        rows = conn.execute(
            """
            SELECT outcome, COUNT(*) AS count
            FROM signals
            WHERE chat_id = ?
              AND signal IN ('LONG', 'SHORT')
              AND outcome IS NOT NULL
            GROUP BY outcome
            """,
            (chat_id,)
        ).fetchall()

        conn.close()

    wins = 0
    losses = 0
    invalid = 0

    for row in rows:

        if row["outcome"] == "WIN":
            wins = row["count"]

        elif row["outcome"] == "LOSS":
            losses = row["count"]

        elif row["outcome"] == "INVALID":
            invalid = row["count"]

    decided = wins + losses

    if decided > 0:
        win_rate = wins / decided * 100
    else:
        win_rate = 0

    await update.message.reply_text(
        f"""
📊 PAPER PERFORMANCE

WIN: {wins}
LOSS: {losses}
INVALID: {invalid}

Decided Trades: {decided}

Recorded Win Rate:
{win_rate:.2f}%

This is based only on manually recorded results.
It is NOT a guarantee of future performance.
""".strip()
    )


# ============================================================
# /HISTORY
# ============================================================

async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    chat_id = update.effective_chat.id

    with db_lock:

        conn = get_db()

        rows = conn.execute(
            """
            SELECT
                id,
                created_at,
                symbol,
                timeframe,
                signal,
                confidence,
                setup_score,
                outcome
            FROM signals
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (chat_id,)
        ).fetchall()

        conn.close()

    if not rows:

        await update.message.reply_text(
            "No signal history yet."
        )

        return

    lines = [
        "📜 LAST 10 SIGNALS",
        ""
    ]

    for row in rows:

        outcome = row["outcome"] or "-"

        lines.append(
            f"#{row['id']} | "
            f"{row['symbol']} | "
            f"{row['timeframe']} | "
            f"{row['signal']} | "
            f"C:{row['confidence']}% | "
            f"S:{row['setup_score']} | "
            f"{outcome}"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# /CLEAR
# ============================================================

async def clear_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    chat_id = update.effective_chat.id

    pending_multi[chat_id].clear()

    multi_mode[chat_id] = False

    await update.message.reply_text(
        "🧹 Pending screenshots cleared."
    )


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    print(
        "Telegram error:",
        repr(context.error)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    init_db()

    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True
    )

    flask_thread.start()

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start_command
        )
    )

    application.add_handler(
        CommandHandler(
            "help",
            help_command
        )
    )

    application.add_handler(
        CommandHandler(
            "status",
            status_command
        )
    )

    application.add_handler(
        CommandHandler(
            "multi",
            multi_command
        )
    )

    application.add_handler(
        CommandHandler(
            "single",
            single_command
        )
    )

    application.add_handler(
        CommandHandler(
            "analyze",
            analyze_command
        )
    )

    application.add_handler(
        CommandHandler(
            "result",
            result_command
        )
    )

    application.add_handler(
        CommandHandler(
            "stats",
            stats_command
        )
    )

    application.add_handler(
        CommandHandler(
            "history",
            history_command
        )
    )

    application.add_handler(
        CommandHandler(
            "clear",
            clear_command
        )
    )

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler
        )
    )

    application.add_error_handler(
        error_handler
    )

    print(
        "🚀 Binance Futures AI Assistant V3 started"
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()