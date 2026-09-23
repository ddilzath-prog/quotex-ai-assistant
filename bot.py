import os
import base64
import json
import threading
import logging

from flask import Flask, jsonify
from openai import OpenAI

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


# =========================================================
# CONFIG
# =========================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

# Directional signal requires this minimum evidence score.
MIN_CONFIDENCE = 80

MAX_IMAGE_BYTES = 8 * 1024 * 1024


if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable is missing.")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is missing.")


client = OpenAI(api_key=OPENAI_API_KEY)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("quotex-ai")


# =========================================================
# RENDER HEALTH SERVER
# =========================================================

app = Flask(__name__)


@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "bot": "Quotex AI Assistant"
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


def run_web_server():
    port = int(os.getenv("PORT", "10000"))

    app.run(
        host="0.0.0.0",
        port=port
    )


# =========================================================
# MASTER ANALYSIS PROMPT
# =========================================================

SYSTEM_PROMPT = r"""
You are the visual analysis engine of Quotex AI Assistant.

Your job is to analyze a trading-chart SCREENSHOT conservatively.

IMPORTANT:
You only know what is visibly present in the image.

NEVER:
- invent unseen candles
- invent prices
- invent indicators
- invent market data
- pretend to know hidden broker data
- claim guaranteed accuracy
- force a signal when evidence is weak

A screenshot can never reveal hidden OTC/broker algorithms.

For OTC charts, analyze ONLY observable price behaviour.

=========================================================
STEP 0 — SCREENSHOT QUALITY
=========================================================

First inspect the image.

Check:

1. Are candles clearly visible?
2. Is the latest candle identifiable?
3. Is enough previous history visible?
4. Is the chart blurred?
5. Is the chart cropped?
6. Are important candles hidden?
7. Are indicators/UI elements blocking the chart?

If chart quality is poor:

signal = NO TRADE

Do not guess.

=========================================================
STEP 1 — MARKET STRUCTURE
=========================================================

Analyze multiple visible candles.

Identify:

- Higher High
- Higher Low
- Lower High
- Lower Low
- Bullish structure
- Bearish structure
- Sideways/range
- Breakout
- Breakdown
- Structure failure
- Possible reversal

Never determine the complete trend from one candle.

=========================================================
STEP 2 — CANDLE PSYCHOLOGY
=========================================================

Analyze:

- candle body size
- bullish/bearish body
- upper wick
- lower wick
- rejection
- indecision
- doji-like behaviour
- engulfing behaviour
- momentum candles
- exhaustion candles
- consecutive candles
- weakening candles

Interpret candle psychology using surrounding candles.

ONE candle alone is NOT sufficient for high confidence.

=========================================================
STEP 3 — MOMENTUM
=========================================================

Determine:

- bullish momentum
- bearish momentum
- neutral momentum
- increasing momentum
- decreasing momentum
- exhaustion

Compare recent candles with earlier visible candles.

A large candle does NOT automatically mean strong continuation.

=========================================================
STEP 4 — SUPPORT AND RESISTANCE
=========================================================

Identify visible reaction areas.

Check:

- support
- resistance
- repeated rejection
- breakout
- breakdown
- retest
- failed breakout
- failed breakdown

Do not invent exact levels if the price scale cannot be read.

=========================================================
STEP 5 — TRAP DETECTION
=========================================================

Look for observable signs of:

- fake breakout
- fake breakdown
- breakout failure
- wick rejection
- liquidity-sweep-like behaviour
- late-entry danger
- weak continuation
- reversal trap

If trap evidence exists:

reduce confidence.

=========================================================
STEP 6 — CONTINUATION VS REVERSAL
=========================================================

Classify the setup:

CONTINUATION
REVERSAL
RANGE
UNCLEAR

Continuation needs structure + momentum confirmation.

Reversal needs stronger evidence than a single wick or candle.

=========================================================
STEP 7 — CURRENT CANDLE
=========================================================

Analyze the latest visible candle.

Check:

- direction
- body strength
- wick structure
- location
- relation to previous candle
- relation to support/resistance
- confirmation or contradiction

If the candle appears to still be forming:

treat it as LOWER reliability.

=========================================================
STEP 8 — OTC VISIBLE PRICE BEHAVIOUR
=========================================================

If the chart is identified as OTC:

Analyze only visible behaviour such as:

- repeated short-term rejection
- repeated reaction zones
- rapid reversals
- range behaviour
- unusual wick clustering
- breakout failures
- momentum changes
- repeated candle reactions

DO NOT claim knowledge of:

- Quotex's hidden algorithm
- broker manipulation
- future OTC price generation
- hidden order flow
- hidden liquidity
- unseen market data

Those are not visible from a screenshot.

=========================================================
STEP 9 — EIGHT-CONFIRMATION DECISION
=========================================================

Evaluate:

1. Market structure
2. Candle psychology
3. Momentum
4. Support/resistance
5. Trap risk
6. Continuation/reversal
7. Current candle confirmation
8. Chart quality / OTC visible behaviour

Evidence must be internally consistent.

If important components conflict:

NO TRADE.

=========================================================
STRICT SIGNAL RULE
=========================================================

Possible signals:

UP
DOWN
NO TRADE

Only output UP when:

- evidence strongly supports bullish direction
- multiple independent confirmations agree
- trap risk is acceptable
- current candle does not contradict the setup
- chart quality is sufficient
- confidence >= 80

Only output DOWN when:

- evidence strongly supports bearish direction
- multiple independent confirmations agree
- trap risk is acceptable
- current candle does not contradict the setup
- chart quality is sufficient
- confidence >= 80

Otherwise:

NO TRADE

=========================================================
IMPORTANT
=========================================================

Confidence is an evidence-strength score.

It is NOT a guaranteed probability of winning.

Do not artificially increase confidence to produce a signal.

If uncertain, choose NO TRADE.

=========================================================
OUTPUT
=========================================================

Return ONLY valid JSON.

Use exactly this structure:

{
  "signal": "UP",
  "confidence": 84,
  "up_score": 84,
  "down_score": 16,
  "market_structure": "Bullish",
  "candle_psychology": "Bullish continuation",
  "momentum": "Strong",
  "support": "Confirmed",
  "resistance": "Weak",
  "trap_risk": "Low",
  "setup": "Continuation",
  "otc_behavior": "Visible bullish rejection/continuation",
  "chart_quality": "Good",
  "confirmation": "Strong",
  "reason": "Multiple independent confirmations agree."
}

For NO TRADE:

{
  "signal": "NO TRADE",
  "confidence": 48,
  "up_score": 48,
  "down_score": 52,
  "market_structure": "Unclear",
  "candle_psychology": "Indecision",
  "momentum": "Weak",
  "support": "Unclear",
  "resistance": "Unclear",
  "trap_risk": "High",
  "setup": "Unclear",
  "otc_behavior": "Mixed",
  "chart_quality": "Fair",
  "confirmation": "Weak",
  "reason": "Evidence is conflicting."
}

Rules:

- up_score + down_score MUST equal 100.
- signal must be UP, DOWN, or NO TRADE.
- confidence must equal the stronger directional score.
- Never use markdown.
- Never write text outside JSON.
"""


# =========================================================
# ANALYZE CHART
# =========================================================

def analyze_chart(image_bytes: bytes) -> dict:

    if not image_bytes:
        return {
            "signal": "NO TRADE",
            "confidence": 0,
            "reason": "Empty image."
        }

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return {
            "signal": "NO TRADE",
            "confidence": 0,
            "reason": "Screenshot is too large."
        }

    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    response = client.responses.create(
        model=MODEL,
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
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Analyze this trading chart screenshot "
                            "using every required analysis layer. "
                            "Do not guess unseen information."
                        )
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            "data:image/jpeg;base64,"
                            + image_base64
                        )
                    }
                ]
            }
        ]
    )

    raw = response.output_text.strip()

    # Remove accidental markdown fences.
    if raw.startswith("```"):
        raw = raw.replace("```json", "")
        raw = raw.replace("```", "")
        raw = raw.strip()

    try:
        result = json.loads(raw)

    except Exception:

        logger.error(
            "Invalid JSON returned by model: %s",
            raw
        )

        return {
            "signal": "NO TRADE",
            "confidence": 0,
            "reason": "Invalid analysis response."
        }

    # =====================================================
    # HARD VALIDATION
    # =====================================================

    signal = str(
        result.get("signal", "NO TRADE")
    ).upper()

    if signal not in {
        "UP",
        "DOWN",
        "NO TRADE"
    }:
        signal = "NO TRADE"

    try:
        up_score = float(
            result.get("up_score", 0)
        )

        down_score = float(
            result.get("down_score", 0)
        )

    except Exception:

        up_score = 0
        down_score = 0

    # Normalize score.
    if up_score < 0:
        up_score = 0

    if down_score < 0:
        down_score = 0

    total = up_score + down_score

    if total <= 0:
        up_score = 50
        down_score = 50
    else:
        up_score = (up_score / total) * 100
        down_score = (down_score / total) * 100

    up_score = round(up_score, 1)
    down_score = round(
        100 - up_score,
        1
    )

    confidence = round(
        max(up_score, down_score),
        1
    )

    # =====================================================
    # FINAL HARD SIGNAL GATE
    # =====================================================

    if signal == "UP":

        if up_score < MIN_CONFIDENCE:
            signal = "NO TRADE"

    elif signal == "DOWN":

        if down_score < MIN_CONFIDENCE:
            signal = "NO TRADE"

    else:

        signal = "NO TRADE"

    # If NO TRADE, preserve evidence confidence,
    # but never convert it into a directional signal.
    result["signal"] = signal
    result["up_score"] = up_score
    result["down_score"] = down_score
    result["confidence"] = confidence

    return result


# =========================================================
# TELEGRAM FORMAT
# =========================================================

def format_result(result: dict) -> str:

    signal = result.get(
        "signal",
        "NO TRADE"
    )

    confidence = result.get(
        "confidence",
        0
    )

    up_score = result.get(
        "up_score",
        0
    )

    down_score = result.get(
        "down_score",
        0
    )

    trend = result.get(
        "market_structure",
        "Unclear"
    )

    candle = result.get(
        "candle_psychology",
        "Unclear"
    )

    momentum = result.get(
        "momentum",
        "Unclear"
    )

    support = result.get(
        "support",
        "Unclear"
    )

    resistance = result.get(
        "resistance",
        "Unclear"
    )

    trap = result.get(
        "trap_risk",
        "Unknown"
    )

    setup = result.get(
        "setup",
        "Unclear"
    )

    otc = result.get(
        "otc_behavior",
        "Unknown"
    )

    quality = result.get(
        "chart_quality",
        "Unknown"
    )

    confirmation = result.get(
        "confirmation",
        "Weak"
    )

    reason = result.get(
        "reason",
        ""
    )

    if signal == "UP":
        icon = "🟢"

    elif signal == "DOWN":
        icon = "🔴"

    else:
        icon = "⚪"

    return (
        "🤖 QUOTEX AI ASSISTANT\n"
        "\n"
        f"{icon} NEXT CANDLE: {signal}\n"
        f"UP: {up_score}%\n"
        f"DOWN: {down_score}%\n"
        f"CONFIDENCE: {confidence}%\n"
        "\n"
        f"TREND: {trend}\n"
        f"CANDLE PSYCHOLOGY: {candle}\n"
        f"MOMENTUM: {momentum}\n"
        f"SUPPORT: {support}\n"
        f"RESISTANCE: {resistance}\n"
        f"TRAP RISK: {trap}\n"
        f"SETUP: {setup}\n"
        f"OTC BEHAVIOUR: {otc}\n"
        f"CHART QUALITY: {quality}\n"
        f"CONFIRMATION: {confirmation}\n"
        "\n"
        f"REASON: {reason}\n"
        "\n"
        "⚠️ Analysis only. No guaranteed outcome."
    )


# =========================================================
# START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🤖 QUOTEX AI ASSISTANT\n\n"
        "Send a clear trading-chart screenshot.\n\n"
        "I analyze:\n"
        "1. Market Structure\n"
        "2. Candle Psychology\n"
        "3. Momentum\n"
        "4. Support / Resistance\n"
        "5. Trap Detection\n"
        "6. Continuation / Reversal\n"
        "7. Current Candle\n"
        "8. OTC Visible Behaviour\n\n"
        "Weak confirmation → NO TRADE\n"
        "Minimum directional threshold → 80%"
    )


# =========================================================
# PHOTO HANDLER
# =========================================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message:
        return

    if not update.message.photo:
        return

    status_message = await update.message.reply_text(
        "🔍 ANALYZING...\n\n"
        "Structure → Candle Psychology → "
        "Momentum → S/R → Trap → OTC → Confirmation"
    )

    try:

        photo = update.message.photo[-1]

        telegram_file = await photo.get_file()

        image_bytes = bytes(
            await telegram_file.download_as_bytearray()
        )

        result = analyze_chart(
            image_bytes
        )

        message = format_result(
            result
        )

        await status_message.edit_text(
            message
        )

    except Exception as e:

        logger.exception(
            "Chart analysis failed"
        )

        await status_message.edit_text(
            "❌ ANALYSIS ERROR\n\n"
            "Please send a clear chart screenshot again."
        )


# =========================================================
# TEXT HANDLER
# =========================================================

async def handle_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message:
        return

    await update.message.reply_text(
        "📸 Please send a trading-chart screenshot."
    )


# =========================================================
# ERROR HANDLER
# =========================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    logger.error(
        "Telegram error: %r",
        context.error
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("======================================")
    print("QUOTEX AI ASSISTANT")
    print("Starting...")
    print("======================================")

    # ---------------------------------
    # Render health server
    # ---------------------------------

    server_thread = threading.Thread(
        target=run_web_server,
        daemon=True
    )

    server_thread.start()

    # ---------------------------------
    # Telegram
    # ---------------------------------

    telegram_app = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    telegram_app.add_handler(
        CommandHandler(
            "start",
            start
        )
    )

    telegram_app.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo
        )
    )

    telegram_app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_text
        )
    )

    telegram_app.add_error_handler(
        error_handler
    )

    print("Telegram bot started.")
    print("Model:", MODEL)
    print(
        "Minimum confidence:",
        MIN_CONFIDENCE
    )

    telegram_app.run_polling(
        drop_pending_updates=True
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()





