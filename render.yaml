import os
import base64
import json
import re
import threading
from io import BytesIO
from collections import defaultdict, deque

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

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-mini"
)

MIN_CONFIDENCE = int(
    os.getenv("MIN_CONFIDENCE", "80")
)

MAX_HISTORY = int(
    os.getenv("MAX_HISTORY", "5")
)

MAX_IMAGE_BYTES = 8 * 1024 * 1024


if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN environment variable is missing."
    )

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is missing."
    )


client = OpenAI(
    api_key=OPENAI_API_KEY
)


# =========================================================
# ANALYSIS MEMORY
# =========================================================

chat_history = defaultdict(
    lambda: deque(
        maxlen=MAX_HISTORY
    )
)


# =========================================================
# RENDER HEALTH SERVER
# =========================================================

app = Flask(__name__)


@app.get("/")
def home():

    return jsonify({
        "status": "online",
        "bot": "Binance Futures AI Assistant",
        "version": "1.0",
        "mode": "Screenshot Analysis"
    })


@app.get("/health")
def health():

    return jsonify({
        "status": "healthy"
    })


@app.get("/status")
def status():

    return jsonify({
        "status": "running",
        "model": OPENAI_MODEL,
        "confidence_threshold": MIN_CONFIDENCE,
        "history": MAX_HISTORY
    })


def run_web_server():

    port = int(
        os.getenv("PORT", "10000")
    )

    app.run(
        host="0.0.0.0",
        port=port
    )


# =========================================================
# BINANCE FUTURES AI SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = f"""
You are BINANCE FUTURES AI ASSISTANT.

Your task is to analyze a Binance Futures trading
chart screenshot.

This is a SCREENSHOT-BASED analysis system.

Do NOT place trades.

Do NOT claim guaranteed profit.

Do NOT claim 99% or 100% accuracy.

The purpose is to identify high-quality setups
and reject weak setups.

QUALITY > QUANTITY.

If the chart does not provide enough confirmation:

NO TRADE.

========================================================
SCREENSHOT ANALYSIS
========================================================

Analyze the ENTIRE visible screenshot.

Do not analyze only the latest candle.

Read all visible information including:

- Trading pair
- Futures / Perpetual
- Current price
- Timeframe
- Candles
- Moving averages
- Volume
- Visible order book
- Long/Short ratio if visible
- Support
- Resistance
- Previous highs
- Previous lows
- Breakout areas
- Rejection areas

Only use information actually visible.

If something is not visible:

mark it UNKNOWN.

Never invent data.

========================================================
1. MARKET STRUCTURE
========================================================

Analyze:

Higher High
Higher Low
Lower High
Lower Low

Identify:

BULLISH
BEARISH
RANGE
UNCLEAR

Determine:

STRONG
MODERATE
WEAK
UNCLEAR

========================================================
2. CANDLE PSYCHOLOGY
========================================================

Analyze:

- candle body size
- upper wick
- lower wick
- rejection
- consecutive candles
- engulfing behaviour
- momentum candles
- indecision candles
- compression
- expansion
- failed continuation
- reversal behaviour

Do not make a decision from one candle alone.

========================================================
3. MOMENTUM
========================================================

Check:

- directional candle sequence
- body expansion
- body contraction
- acceleration
- deceleration
- opposite candle pressure
- momentum loss

Classify:

STRONG
MODERATE
WEAK
UNCLEAR

========================================================
4. SUPPORT
========================================================

Find visible support zones.

Check:

- previous reactions
- multiple touches
- rejection
- breakdown
- retest

Classify:

CONFIRMED
WEAK
UNCLEAR

========================================================
5. RESISTANCE
========================================================

Find visible resistance zones.

Check:

- previous reactions
- multiple touches
- rejection
- breakout
- retest

Classify:

CONFIRMED
WEAK
UNCLEAR

========================================================
6. BREAKOUT
========================================================

Determine:

CONFIRMED BREAKOUT
WEAK BREAKOUT
FALSE BREAKOUT
NO BREAKOUT
UNCLEAR

A breakout should have supporting price action.

========================================================
7. TRAP PSYCHOLOGY
========================================================

Look for:

- Bull Trap
- Bear Trap
- False Breakout
- Liquidity-sweep-like behaviour
- Strong rejection

Do not claim actual broker liquidity.

These are chart interpretations only.

========================================================
8. EXHAUSTION
========================================================

Check for:

- large candles
- long wicks
- shrinking bodies
- repeated rejection
- failure to continue
- support/resistance pressure
- opposite candle pressure

Classify:

HIGH
MEDIUM
LOW
UNCLEAR

========================================================
9. CONTINUATION
========================================================

Determine:

LIKELY
UNLIKELY
UNCLEAR

Use:

- market structure
- candle psychology
- momentum
- support/resistance
- breakout confirmation
- trap risk

========================================================
10. REVERSAL
========================================================

A reversal should require multiple confirmations.

Possible confirmations:

- strong rejection
- momentum weakening
- structure change
- failed continuation
- support/resistance interaction
- opposite candle confirmation

Never call reversal from one wick alone.

========================================================
11. MOVING AVERAGES
========================================================

If MA/EMA is visible:

Analyze:

- price relative to MA
- MA direction
- MA separation
- crossover
- dynamic support/resistance
- compression

Do NOT invent an MA value.

========================================================
12. VOLUME
========================================================

If volume is visible:

Analyze:

- volume expansion
- volume contraction
- breakout volume
- rejection volume
- unusual volume spike

Do not assume volume if it is not visible.

========================================================
13. ORDER BOOK
========================================================

If order book is visible:

Analyze only visible information.

Possible observations:

- bid/ask imbalance
- visible concentration
- pressure near current price

IMPORTANT:

Order-book information can change quickly.

Do not treat it as guaranteed future direction.

If order book is not visible:

UNKNOWN.

========================================================
14. LONG / SHORT RATIO
========================================================

If a Long/Short ratio is visible:

Record it.

Do not assume that a high Long ratio automatically
means SHORT.

Do not assume that a high Short ratio automatically
means LONG.

Use it only as supporting evidence.

========================================================
15. MULTI-CONFIRMATION
========================================================

A directional signal requires agreement between
multiple independent factors.

For LONG:

Prefer:

- bullish structure
- bullish candle psychology
- supportive momentum
- support/retest
- confirmed breakout OR strong rejection
- acceptable volume
- no strong bear trap
- confirmation good/strong

For SHORT:

Prefer:

- bearish structure
- bearish candle psychology
- bearish momentum
- resistance/retest
- confirmed breakdown OR strong rejection
- acceptable volume
- no strong bull trap
- confirmation good/strong

========================================================
16. CONFLICT FILTER
========================================================

If major factors conflict:

NO TRADE.

Examples:

Bullish trend + strong resistance rejection
= caution

Bearish trend + strong support rejection
= caution

Breakout + immediate failure
= caution

Strong momentum + extreme exhaustion
= caution

Unclear structure
= NO TRADE

========================================================
17. CONFIDENCE
========================================================

Return:

LONG probability
SHORT probability

LONG + SHORT MUST equal exactly 100.

CONFIDENCE =
higher of LONG and SHORT.

Minimum directional threshold:

{MIN_CONFIDENCE}%

If:

LONG >= {MIN_CONFIDENCE}

AND confirmations are strong enough:

candidate = LONG

If:

SHORT >= {MIN_CONFIDENCE}

AND confirmations are strong enough:

candidate = SHORT

Otherwise:

NO TRADE

IMPORTANT:

Confidence is an analytical estimate.

It is NOT a guaranteed win probability.

========================================================
18. RISK
========================================================

Classify:

LOW
MEDIUM
HIGH

High risk:

NO TRADE.

========================================================
19. ENTRY / STOP / TARGET
========================================================

If the screenshot provides enough visible price
information:

You may estimate:

ENTRY_ZONE
STOP_LOSS_LEVEL
TAKE_PROFIT_LEVEL

But:

Never invent exact levels.

If levels cannot be reliably determined:

return UNKNOWN.

Use price levels visible in the screenshot.

Do not claim guaranteed targets.

========================================================
20. FINAL DECISION
========================================================

The final signal must be one of:

LONG
SHORT
NO TRADE

Use NO TRADE when:

- confidence < threshold
- structure unclear
- confirmation weak
- risk high
- major factors conflict
- chart visibility is poor
- price location is dangerous
- setup depends on guessing

========================================================
OUTPUT
========================================================

Return ONLY valid JSON.

Use EXACTLY this structure:

{{
    "symbol": "ETHUSDT",
    "market": "BINANCE FUTURES",
    "timeframe": "15m",

    "signal": "LONG | SHORT | NO TRADE",

    "long_probability": 0,
    "short_probability": 0,
    "confidence": 0,

    "trend": "Bullish | Bearish | Range | Unclear",

    "market_structure": "Strong | Moderate | Weak | Unclear",

    "candle_psychology": "Bullish | Bearish | Mixed | Unclear",

    "momentum": "Strong | Moderate | Weak | Unclear",

    "support": "Confirmed | Weak | Unclear",

    "resistance": "Confirmed | Weak | Unclear",

    "breakout": "Confirmed | Weak | False Breakout | None | Unclear",

    "trap": "Bull Trap | Bear Trap | Possible Trap | None | Unclear",

    "continuation": "Likely | Unlikely | Unclear",

    "reversal": "Likely | Unlikely | Unclear",

    "exhaustion": "High | Medium | Low | Unclear",

    "volume": "Strong | Normal | Weak | Unclear",

    "order_book": "Bullish | Bearish | Balanced | Unclear",

    "long_short_ratio": "Bullish Bias | Bearish Bias | Balanced | Unavailable",

    "confirmation": "Strong | Good | Weak | Unclear",

    "risk": "Low | Medium | High",

    "entry_zone": "text or UNKNOWN",

    "stop_loss": "text or UNKNOWN",

    "take_profit": "text or UNKNOWN",

    "reason": "Short factual explanation"
}}

Probabilities MUST be integers.

LONG + SHORT must equal exactly 100.

Never output markdown.

Never output anything outside JSON.
"""


# =========================================================
# IMAGE PREPARATION
# =========================================================

def prepare_image(
    image_bytes: bytes
) -> bytes:

    if len(image_bytes) <= MAX_IMAGE_BYTES:
        return image_bytes

    try:

        from PIL import Image

        image = Image.open(
            BytesIO(image_bytes)
        )

        image.thumbnail(
            (2400, 2400)
        )

        output = BytesIO()

        image.convert("RGB").save(
            output,
            format="JPEG",
            quality=88,
            optimize=True
        )

        return output.getvalue()

    except Exception as error:

        print(
            "IMAGE PREPARATION ERROR:",
            repr(error)
        )

        return image_bytes


# =========================================================
# JSON PARSER
# =========================================================

def extract_json(
    text: str
) -> dict:

    text = text.strip()

    try:

        return json.loads(text)

    except Exception:

        pass


    cleaned = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    cleaned = cleaned.replace(
        "```",
        ""
    ).strip()


    try:

        return json.loads(
            cleaned
        )

    except Exception:

        pass


    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL
    )


    if match:

        try:

            return json.loads(
                match.group(0)
            )

        except Exception:

            pass


    raise ValueError(
        "AI did not return valid JSON."
    )


# =========================================================
# SAFE INTEGER
# =========================================================

def safe_int(
    value,
    default=0
):

    try:

        return int(
            float(value)
        )

    except Exception:

        return default


# =========================================================
# NORMALIZE PROBABILITIES
# =========================================================

def normalize_probabilities(
    long_probability,
    short_probability
):

    long_probability = max(
        0,
        min(
            100,
            long_probability
        )
    )

    short_probability = max(
        0,
        min(
            100,
            short_probability
        )
    )


    total = (
        long_probability
        + short_probability
    )


    if total == 100:

        return (
            long_probability,
            short_probability
        )


    if total <= 0:

        return 50, 50


    long_probability = round(
        (
            long_probability
            / total
        ) * 100
    )


    short_probability = (
        100 - long_probability
    )


    return (
        long_probability,
        short_probability
    )


# =========================================================
# QUALITY GATE
# =========================================================

def quality_gate(
    data: dict
) -> dict:

    long_probability = safe_int(
        data.get(
            "long_probability",
            50
        ),
        50
    )

    short_probability = safe_int(
        data.get(
            "short_probability",
            50
        ),
        50
    )


    long_probability, short_probability = (
        normalize_probabilities(
            long_probability,
            short_probability
        )
    )


    confidence = max(
        long_probability,
        short_probability
    )


    trend = str(
        data.get(
            "trend",
            "Unclear"
        )
    ).lower()


    structure = str(
        data.get(
            "market_structure",
            "Unclear"
        )
    ).lower()


    candle = str(
        data.get(
            "candle_psychology",
            "Unclear"
        )
    ).lower()


    momentum = str(
        data.get(
            "momentum",
            "Unclear"
        )
    ).lower()


    confirmation = str(
        data.get(
            "confirmation",
            "Unclear"
        )
    ).lower()


    risk = str(
        data.get(
            "risk",
            "High"
        )
    ).lower()


    breakout = str(
        data.get(
            "breakout",
            "Unclear"
        )
    ).lower()


    trap = str(
        data.get(
            "trap",
            "Unclear"
        )
    ).lower()


    exhaustion = str(
        data.get(
            "exhaustion",
            "Unclear"
        )
    ).lower()


    reversal = str(
        data.get(
            "reversal",
            "Unclear"
        )
    ).lower()


    force_no_trade = False


    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    if confidence < MIN_CONFIDENCE:

        force_no_trade = True


    # -----------------------------------------------------
    # CONFIRMATION
    # -----------------------------------------------------

    if confirmation in [
        "weak",
        "unclear"
    ]:

        force_no_trade = True


    # -----------------------------------------------------
    # RISK
    # -----------------------------------------------------

    if risk == "high":

        force_no_trade = True


    # -----------------------------------------------------
    # STRUCTURE
    # -----------------------------------------------------

    if structure == "unclear":

        force_no_trade = True


    # -----------------------------------------------------
    # CANDLE
    # -----------------------------------------------------

    if candle in [
        "mixed",
        "unclear"
    ]:

        force_no_trade = True


    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    if momentum == "unclear":

        force_no_trade = True


    # -----------------------------------------------------
    # EXHAUSTION
    # -----------------------------------------------------

    if exhaustion == "high":

        force_no_trade = True


    # -----------------------------------------------------
    # FALSE BREAKOUT
    # -----------------------------------------------------

    if breakout == "false breakout":

        force_no_trade = True


    # -----------------------------------------------------
    # TRAP
    # -----------------------------------------------------

    if trap == "possible trap":

        force_no_trade = True


    # -----------------------------------------------------
    # REVERSAL
    # -----------------------------------------------------

    if reversal == "likely":

        if confirmation != "strong":

            force_no_trade = True


    # -----------------------------------------------------
    # DETERMINE CANDIDATE
    # -----------------------------------------------------

    if long_probability > short_probability:

        candidate = "LONG"

    elif short_probability > long_probability:

        candidate = "SHORT"

    else:

        candidate = "NO TRADE"


    # -----------------------------------------------------
    # DIRECTION CONSISTENCY
    # -----------------------------------------------------

    if candidate == "LONG":

        if trend == "bearish":

            force_no_trade = True

        if candle == "bearish":

            force_no_trade = True


    elif candidate == "SHORT":

        if trend == "bullish":

            force_no_trade = True

        if candle == "bullish":

            force_no_trade = True


    # -----------------------------------------------------
    # FINAL SIGNAL
    # -----------------------------------------------------

    if force_no_trade:

        final_signal = "NO TRADE"

    else:

        if (
            candidate == "LONG"
            and long_probability >= MIN_CONFIDENCE
        ):

            final_signal = "LONG"

        elif (
            candidate == "SHORT"
            and short_probability >= MIN_CONFIDENCE
        ):

            final_signal = "SHORT"

        else:

            final_signal = "NO TRADE"


    data["long_probability"] = (
        long_probability
    )

    data["short_probability"] = (
        short_probability
    )

    data["confidence"] = confidence

    data["signal"] = final_signal


    return data


# =========================================================
# HISTORY CONTEXT
# =========================================================

def build_history_context(
    chat_id
):

    history = chat_history.get(
        chat_id
    )


    if not history:

        return (
            "No previous analysis."
        )


    lines = [
        "Previous analysis context:"
    ]


    for item in history:

        lines.append(
            item
        )


    return "\n".join(
        lines
    )


# =========================================================
# SAVE HISTORY
# =========================================================

def save_history(
    chat_id,
    data
):

    summary = (
        f"{data.get('symbol', 'UNKNOWN')} | "
        f"{data.get('timeframe', 'UNKNOWN')} | "
        f"Signal={data['signal']} | "
        f"LONG={data['long_probability']}% | "
        f"SHORT={data['short_probability']}% | "
        f"Confidence={data['confidence']}% | "
        f"Trend={data.get('trend')} | "
        f"Risk={data.get('risk')}"
    )


    chat_history[
        chat_id
    ].append(summary)


# =========================================================
# FORMAT TELEGRAM MESSAGE
# =========================================================

def format_result(
    data: dict
) -> str:

    signal = data[
        "signal"
    ]


    if signal == "LONG":

        icon = "🟢"

    elif signal == "SHORT":

        icon = "🔴"

    else:

        icon = "⚪"


    return (

        "🤖 BINANCE FUTURES AI\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"

        f"📌 SYMBOL: "
        f"{data.get('symbol', 'UNKNOWN')}\n"

        f"⏱ TIMEFRAME: "
        f"{data.get('timeframe', 'UNKNOWN')}\n\n"

        f"{icon} SIGNAL: "
        f"{signal}\n\n"

        f"🟢 LONG: "
        f"{data['long_probability']}%\n"

        f"🔴 SHORT: "
        f"{data['short_probability']}%\n"

        f"🎯 CONFIDENCE: "
        f"{data['confidence']}%\n\n"

        "📊 MARKET STRUCTURE\n"

        f"Trend: "
        f"{data.get('trend')}\n"

        f"Structure: "
        f"{data.get('market_structure')}\n"

        f"Momentum: "
        f"{data.get('momentum')}\n\n"

        "🕯 CANDLE PSYCHOLOGY\n"

        f"{data.get('candle_psychology')}\n\n"

        "📍 SUPPORT / RESISTANCE\n"

        f"Support: "
        f"{data.get('support')}\n"

        f"Resistance: "
        f"{data.get('resistance')}\n\n"

        "💥 PRICE ACTION\n"

        f"Breakout: "
        f"{data.get('breakout')}\n"

        f"Trap: "
        f"{data.get('trap')}\n"

        f"Continuation: "
        f"{data.get('continuation')}\n"

        f"Reversal: "
        f"{data.get('reversal')}\n"

        f"Exhaustion: "
        f"{data.get('exhaustion')}\n\n"

        "📈 MARKET DATA VISIBLE\n"

        f"Volume: "
        f"{data.get('volume')}\n"

        f"Order Book: "
        f"{data.get('order_book')}\n"

        f"Long/Short: "
        f"{data.get('long_short_ratio')}\n\n"

        "🛡 RISK\n"

        f"Risk: "
        f"{data.get('risk')}\n"

        f"Confirmation: "
        f"{data.get('confirmation')}\n\n"

        "🎯 LEVELS\n"

        f"Entry: "
        f"{data.get('entry_zone')}\n"

        f"Stop Loss: "
        f"{data.get('stop_loss')}\n"

        f"Take Profit: "
        f"{data.get('take_profit')}\n\n"

        f"📝 {data.get('reason')}\n\n"

        "⚠️ Screenshot analysis only. "
        "No guaranteed outcome."
    )


# =========================================================
# ANALYZE IMAGE
# =========================================================

def analyze_chart(
    image_bytes: bytes,
    chat_id=None
):

    image_bytes = prepare_image(
        image_bytes
    )


    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")


    history = (
        build_history_context(
            chat_id
        )
        if chat_id is not None
        else "No previous context."
    )


    user_prompt = f"""
Analyze the ENTIRE Binance Futures screenshot.

The current screenshot is the primary source of truth.

Previous analysis is only historical context.
Do not assume previous predictions were correct.

{history}

Identify the trading pair and timeframe if visible.

Analyze all visible chart information.

Return ONLY the required JSON.
"""


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

                "content": [

                    {
                        "type": "input_text",

                        "text": user_prompt

                    },

                    {
                        "type": "input_image",

                        "image_url": (
                            "data:image/jpeg;base64,"
                            + image_base64
                        ),

                        "detail": "high"
                    }

                ]

            }

        ]

    )


    raw_result = (
        response.output_text.strip()
    )


    print(
        "AI RESULT:",
        raw_result
    )


    data = extract_json(
        raw_result
    )


    data = quality_gate(
        data
    )


    if chat_id is not None:

        save_history(
            chat_id,
            data
        )


    return format_result(
        data
    )


# =========================================================
# START COMMAND
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "🤖 BINANCE FUTURES AI\n\n"

        "Send me a Binance Futures chart "
        "screenshot.\n\n"

        "I will analyze:\n"

        "🕯 Candle Psychology\n"
        "📊 Market Structure\n"
        "📈 Momentum\n"
        "📍 Support / Resistance\n"
        "💥 Breakout / Fake Breakout\n"
        "🪤 Trap Psychology\n"
        "🥵 Exhaustion\n"
        "📊 Volume\n"
        "📖 Order Book if visible\n"
        "⚖️ Long/Short ratio if visible\n\n"

        f"🎯 Minimum confidence: "
        f"{MIN_CONFIDENCE}%\n\n"

        "Weak setup → NO TRADE"
    )


# =========================================================
# HELP COMMAND
# =========================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "📖 BINANCE FUTURES AI\n\n"

        "/start - Start bot\n"
        "/status - Bot status\n"
        "/clear - Clear analysis memory\n"
        "/help - Help\n\n"

        "Send a clear Binance Futures "
        "chart screenshot for analysis."
    )


# =========================================================
# STATUS COMMAND
# =========================================================

async def status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "🟢 BINANCE FUTURES AI ONLINE\n\n"

        f"Model: {OPENAI_MODEL}\n"

        f"Confidence threshold: "
        f"{MIN_CONFIDENCE}%\n"

        f"History: "
        f"{MAX_HISTORY} analyses\n\n"

        "Mode: Screenshot Analysis\n"
        "Trading: DISABLED\n"
        "Quality Filter: ENABLED"
    )


# =========================================================
# CLEAR COMMAND
# =========================================================

async def clear_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.effective_chat:

        return


    chat_id = (
        update.effective_chat.id
    )


    chat_history.pop(
        chat_id,
        None
    )


    await update.message.reply_text(
        "🧹 Analysis history cleared."
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


    if not update.effective_chat:
        return


    chat_id = (
        update.effective_chat.id
    )


    status_message = (
        await update.message.reply_text(

            "🔍 Reading Binance chart...\n\n"

            "🕯 Candle Psychology\n"
            "📊 Market Structure\n"
            "📍 Support / Resistance\n"
            "📈 Momentum\n"
            "🪤 Trap Detection\n"
            "🛡 Risk Filter"
        )
    )


    try:

        photo = (
            update.message.photo[-1]
        )


        telegram_file = (
            await photo.get_file()
        )


        image_bytes = bytes(

            await telegram_file.download_as_bytearray()

        )


        result = analyze_chart(

            image_bytes,

            chat_id

        )


        await status_message.edit_text(
            result
        )


    except Exception as error:

        print(
            "ANALYSIS ERROR:",
            repr(error)
        )


        await status_message.edit_text(

            "❌ Analysis failed.\n\n"

            "Please send a clear Binance Futures "
            "screenshot again."
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

        "📊 Send a Binance Futures "
        "chart screenshot.\n\n"

        "/help\n"
        "/status\n"
        "/clear"
    )


# =========================================================
# ERROR HANDLER
# =========================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    print(
        "TELEGRAM ERROR:",
        repr(context.error)
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "=========================================="
    )

    print(
        "      BINANCE FUTURES AI ASSISTANT"
    )

    print(
        "=========================================="
    )

    print(
        f"MODEL: {OPENAI_MODEL}"
    )

    print(
        f"CONFIDENCE: {MIN_CONFIDENCE}%"
    )

    print(
        "MODE: SCREENSHOT ANALYSIS"
    )

    print(
        "TRADING: DISABLED"
    )


    # -----------------------------------------------------
    # RENDER SERVER
    # -----------------------------------------------------

    server_thread = threading.Thread(

        target=run_web_server,

        daemon=True

    )

    server_thread.start()


    # -----------------------------------------------------
    # TELEGRAM
    # -----------------------------------------------------

    telegram_app = (

        Application.builder()

        .token(BOT_TOKEN)

        .build()

    )


    # -----------------------------------------------------
    # COMMANDS
    # -----------------------------------------------------

    telegram_app.add_handler(

        CommandHandler(
            "start",
            start
        )

    )


    telegram_app.add_handler(

        CommandHandler(
            "help",
            help_command
        )

    )


    telegram_app.add_handler(

        CommandHandler(
            "status",
            status_command
        )

    )


    telegram_app.add_handler(

        CommandHandler(
            "clear",
            clear_command
        )

    )


    # -----------------------------------------------------
    # PHOTO
    # -----------------------------------------------------

    telegram_app.add_handler(

        MessageHandler(

            filters.PHOTO,

            handle_photo

        )

    )


    # -----------------------------------------------------
    # TEXT
    # -----------------------------------------------------

    telegram_app.add_handler(

        MessageHandler(

            filters.TEXT
            & ~filters.COMMAND,

            handle_text

        )

    )


    # -----------------------------------------------------
    # ERRORS
    # -----------------------------------------------------

    telegram_app.add_error_handler(
        error_handler
    )


    print(
        "Telegram bot started."
    )

    print(
        "BINANCE FUTURES AI is ready."
    )


    telegram_app.run_polling(

        drop_pending_updates=True

    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()
