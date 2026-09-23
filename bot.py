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

# You can change this from Render environment variables.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# Minimum confidence required for a directional signal.
MIN_CONFIDENCE = int(
    os.getenv("MIN_CONFIDENCE", "80")
)

# Number of previous textual analyses remembered per chat.
MAX_HISTORY = int(
    os.getenv("MAX_HISTORY", "5")
)

# Maximum image size before compression.
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
# CHAT MEMORY
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
        "bot": "Quotex AI Assistant",
        "version": "3.0",
        "engine": "Multi-Layer Trading Psychology"
    })


@app.get("/health")
def health():

    return jsonify({
        "status": "healthy"
    })


@app.get("/status")
def server_status():

    return jsonify({
        "status": "running",
        "model": OPENAI_MODEL,
        "confidence_threshold": MIN_CONFIDENCE,
        "history_per_chat": MAX_HISTORY
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
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = f"""
You are QUOTEX AI ASSISTANT V3.

You are a screenshot-based trading chart analysis assistant.

Your job is NOT to guarantee profit.

Your job is to identify whether the visible chart provides
enough evidence for a directional next-candle hypothesis.

QUALITY IS MORE IMPORTANT THAN SIGNAL FREQUENCY.

If evidence conflicts, return NO TRADE.

============================================================
CORE PRINCIPLE
============================================================

NEVER invent information.

Only use information visible in the chart.

Do NOT invent:

- candles
- prices
- volume
- indicators
- timeframes
- support levels
- resistance levels
- broker data
- liquidity data
- future candles

If something cannot be seen clearly:

mark it UNKNOWN / UNCLEAR.

============================================================
ANALYSIS LAYERS
============================================================

Analyze all visible candles, not only the final candle.

1. Market Structure
2. Candle Psychology
3. Momentum
4. Support
5. Resistance
6. Breakout
7. False Breakout
8. Trap Behaviour
9. Continuation
10. Exhaustion
11. Reversal
12. OTC caution
13. Final confirmation

============================================================
1. MARKET STRUCTURE
============================================================

Look for:

Higher High
Higher Low
Lower High
Lower Low

Classify:

Bullish
Bearish
Range
Unclear

Check whether structure is:

Strong
Moderate
Weak
Unclear

Do not force a trend.

============================================================
2. CANDLE PSYCHOLOGY
============================================================

Analyze:

- candle body
- upper wick
- lower wick
- rejection
- consecutive candles
- engulfing
- momentum candles
- indecision
- compression
- expansion
- failed continuation
- reversal candles

A single candle is NEVER enough by itself.

Interpret the candle in context.

============================================================
3. MOMENTUM
============================================================

Determine:

Strong
Moderate
Weak
Unclear

Check:

- consecutive directional candles
- candle body expansion
- candle body contraction
- opposite candle pressure
- momentum loss
- acceleration
- deceleration

============================================================
4. SUPPORT / RESISTANCE
============================================================

Look for visible reaction zones.

Check:

- multiple reactions
- previous rejection
- breakout
- retest
- failed retest
- proximity to current candle

Classify:

Confirmed
Weak
Unclear

Never invent exact levels.

============================================================
5. BREAKOUT ANALYSIS
============================================================

Determine whether the latest movement resembles:

Confirmed Breakout
Weak Breakout
False Breakout
Retest
Failed Retest
No Breakout
Unclear

A breakout needs supporting price action.

============================================================
6. TRAP PSYCHOLOGY
============================================================

Look for possible:

Bull Trap
Bear Trap
False Breakout
Liquidity Sweep-like behaviour
Rejection

These are chart interpretations only.

Do not claim knowledge of actual broker liquidity.

============================================================
7. CONTINUATION
============================================================

Check whether the current movement has evidence for continuation.

Consider:

- structure
- momentum
- candle consistency
- S/R location
- breakout confirmation
- opposite candle pressure

Classify:

Likely
Unlikely
Unclear

============================================================
8. EXHAUSTION
============================================================

Check for:

- unusually large candles
- long rejection wicks
- repeated failure to continue
- shrinking candle bodies
- opposite pressure
- resistance/support interaction

Classify:

High
Medium
Low
Unclear

============================================================
9. REVERSAL
============================================================

Do NOT call a reversal from one wick.

Prefer multiple confirmations:

- rejection
- momentum weakening
- structure change
- failed continuation
- S/R interaction
- opposite candle confirmation

============================================================
10. OTC MARKET
============================================================

If chart appears to be OTC:

Apply stricter filtering.

Do NOT assume OTC behaves exactly like a centralized market.

If evidence is ambiguous:

NO TRADE.

If OTC cannot be identified:

UNKNOWN.

============================================================
11. PROBABILITY
============================================================

Return:

UP probability
DOWN probability

UP + DOWN MUST equal exactly 100.

CONFIDENCE = higher of UP and DOWN.

However, probability alone does NOT create a signal.

============================================================
12. FINAL SIGNAL RULE
============================================================

Minimum confidence:

{MIN_CONFIDENCE}%

If UP >= {MIN_CONFIDENCE}:
possible UP.

If DOWN >= {MIN_CONFIDENCE}:
possible DOWN.

But a directional signal is allowed ONLY when:

- structure supports it
- candle psychology supports it
- momentum is not strongly contradictory
- S/R context is acceptable
- trap risk is not strongly contradictory
- continuation/reversal logic is coherent
- confirmation is Strong or Good
- risk is not High

Otherwise:

NO TRADE.

============================================================
13. QUALITY SCORING
============================================================

Internally evaluate:

STRUCTURE
CANDLE
MOMENTUM
S/R
BREAKOUT
TRAP
CONTINUATION
EXHAUSTION
CONFIRMATION

Do not give a directional signal just because one category is strong.

Multiple independent confirmations are required.

============================================================
14. CONFLICT RULE
============================================================

If:

trend bullish
but resistance rejection is strong

or:

trend bearish
but support rejection is strong

or:

breakout occurs but immediately fails

or:

momentum and candle psychology disagree

then prefer:

NO TRADE.

============================================================
15. OUTPUT
============================================================

Return ONLY JSON.

Use exactly this structure:

{{
    "next_candle": "UP | DOWN | NO TRADE",

    "up_probability": 0,
    "down_probability": 0,
    "confidence": 0,

    "market_type": "NORMAL | OTC | UNKNOWN",

    "trend": "Bullish | Bearish | Range | Unclear",

    "market_structure": "Strong | Moderate | Weak | Unclear",

    "candle_psychology": "Bullish | Bearish | Mixed | Unclear",

    "momentum": "Strong | Moderate | Weak | Unclear",

    "support": "Confirmed | Weak | Unclear",

    "resistance": "Confirmed | Weak | Unclear",

    "breakout": "Confirmed | Weak | False Breakout | None | Unclear",

    "trap": "Bull Trap | Bear Trap | Possible Trap | None | Unclear",

    "continuation": "Likely | Unlikely | Unclear",

    "exhaustion": "High | Medium | Low | Unclear",

    "reversal": "Likely | Unlikely | Unclear",

    "confirmation": "Strong | Good | Weak | Unclear",

    "risk_level": "Low | Medium | High",

    "reason": "Short factual explanation"
}}

All probabilities must be integers.

UP + DOWN must equal 100.

Do not include markdown.

Do not include additional text.

Never claim guaranteed accuracy.

Never claim guaranteed profit.
"""


# =========================================================
# IMAGE PREPARATION
# =========================================================

def prepare_image(image_bytes: bytes) -> bytes:

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
            "IMAGE COMPRESSION ERROR:",
            repr(error)
        )

        return image_bytes


# =========================================================
# JSON EXTRACTION
# =========================================================

def extract_json(text: str) -> dict:

    text = text.strip()

    # Direct JSON
    try:
        return json.loads(text)

    except Exception:
        pass

    # Remove markdown fences
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
        return json.loads(cleaned)

    except Exception:
        pass

    # Find JSON object
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
        "AI response is not valid JSON."
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
    up,
    down
):

    up = max(
        0,
        min(100, up)
    )

    down = max(
        0,
        min(100, down)
    )

    total = up + down

    if total == 100:
        return up, down

    if total <= 0:
        return 50, 50

    up_normalized = round(
        (up / total) * 100
    )

    down_normalized = (
        100 - up_normalized
    )

    return (
        up_normalized,
        down_normalized
    )


# =========================================================
# PYTHON QUALITY GATE
# =========================================================

def quality_gate(data: dict) -> dict:

    up = safe_int(
        data.get(
            "up_probability",
            50
        ),
        50
    )

    down = safe_int(
        data.get(
            "down_probability",
            50
        ),
        50
    )

    up, down = normalize_probabilities(
        up,
        down
    )

    confidence = max(
        up,
        down
    )

    requested_signal = str(
        data.get(
            "next_candle",
            "NO TRADE"
        )
    ).upper().strip()

    confirmation = str(
        data.get(
            "confirmation",
            "Unclear"
        )
    ).lower().strip()

    risk = str(
        data.get(
            "risk_level",
            "High"
        )
    ).lower().strip()

    structure = str(
        data.get(
            "market_structure",
            "Unclear"
        )
    ).lower().strip()

    trend = str(
        data.get(
            "trend",
            "Unclear"
        )
    ).lower().strip()

    momentum = str(
        data.get(
            "momentum",
            "Unclear"
        )
    ).lower().strip()

    candle = str(
        data.get(
            "candle_psychology",
            "Unclear"
        )
    ).lower().strip()

    support = str(
        data.get(
            "support",
            "Unclear"
        )
    ).lower().strip()

    resistance = str(
        data.get(
            "resistance",
            "Unclear"
        )
    ).lower().strip()

    breakout = str(
        data.get(
            "breakout",
            "Unclear"
        )
    ).lower().strip()

    trap = str(
        data.get(
            "trap",
            "Unclear"
        )
    ).lower().strip()

    continuation = str(
        data.get(
            "continuation",
            "Unclear"
        )
    ).lower().strip()

    reversal = str(
        data.get(
            "reversal",
            "Unclear"
        )
    ).lower().strip()

    exhaustion = str(
        data.get(
            "exhaustion",
            "Unclear"
        )
    ).lower().strip()


    # =====================================================
    # HARD REJECTIONS
    # =====================================================

    force_no_trade = False

    if confidence < MIN_CONFIDENCE:
        force_no_trade = True

    if confirmation in [
        "weak",
        "unclear"
    ]:
        force_no_trade = True

    if risk == "high":
        force_no_trade = True

    if structure == "unclear":
        force_no_trade = True

    if trend == "unclear":
        force_no_trade = True

    if candle == "mixed":
        force_no_trade = True

    if candle == "unclear":
        force_no_trade = True

    if momentum == "unclear":
        force_no_trade = True

    if continuation == "unclear":
        force_no_trade = True

    # Strong exhaustion = additional caution.
    if exhaustion == "high":
        force_no_trade = True

    # Possible trap is not automatically a trade.
    if trap == "possible trap":
        force_no_trade = True

    # False breakout requires additional confirmation.
    if breakout == "false breakout":
        force_no_trade = True


    # =====================================================
    # DIRECTION CHECK
    # =====================================================

    if up > down:

        candidate = "UP"

    elif down > up:

        candidate = "DOWN"

    else:

        candidate = "NO TRADE"


    # =====================================================
    # DIRECTION CONSISTENCY
    # =====================================================

    if candidate == "UP":

        if trend == "bearish":
            force_no_trade = True

        if candle == "bearish":
            force_no_trade = True

        if momentum == "weak":
            force_no_trade = True

        if trap == "bear trap":
            pass

    elif candidate == "DOWN":

        if trend == "bullish":
            force_no_trade = True

        if candle == "bullish":
            force_no_trade = True

        if momentum == "weak":
            force_no_trade = True

        if trap == "bull trap":
            pass


    # =====================================================
    # SUPPORT / RESISTANCE CAUTION
    # =====================================================

    # If both S/R are unclear, confidence is not trustworthy.
    if (
        support == "unclear"
        and resistance == "unclear"
    ):
        force_no_trade = True


    # =====================================================
    # BREAKOUT CAUTION
    # =====================================================

    if breakout == "weak":
        force_no_trade = True


    # =====================================================
    # REVERSAL CAUTION
    # =====================================================

    if reversal == "likely":

        # A reversal needs strong confirmation.
        if confirmation != "strong":
            force_no_trade = True


    # =====================================================
    # FINAL SIGNAL
    # =====================================================

    if force_no_trade:

        final_signal = "NO TRADE"

    else:

        if candidate == "UP" and up >= MIN_CONFIDENCE:

            final_signal = "UP"

        elif (
            candidate == "DOWN"
            and down >= MIN_CONFIDENCE
        ):

            final_signal = "DOWN"

        else:

            final_signal = "NO TRADE"


    # =====================================================
    # AI REQUEST VS PYTHON RESULT
    # =====================================================

    if requested_signal != final_signal:

        print(
            "QUALITY GATE CHANGED SIGNAL:",
            requested_signal,
            "->",
            final_signal
        )


    data["up_probability"] = up
    data["down_probability"] = down
    data["confidence"] = confidence
    data["next_candle"] = final_signal

    return data


# =========================================================
# FORMAT TELEGRAM
# =========================================================

def format_result(
    data: dict
) -> str:

    signal = data[
        "next_candle"
    ]

    if signal == "UP":

        icon = "🟢"

    elif signal == "DOWN":

        icon = "🔴"

    else:

        icon = "⚪"


    return (

        "🤖 QUOTEX AI ASSISTANT V3\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"

        f"{icon} NEXT CANDLE: {signal}\n\n"

        f"📈 UP: "
        f"{data['up_probability']}%\n"

        f"📉 DOWN: "
        f"{data['down_probability']}%\n"

        f"🎯 CONFIDENCE: "
        f"{data['confidence']}%\n\n"

        "📊 MARKET\n"

        f"Type: "
        f"{data['market_type']}\n"

        f"Trend: "
        f"{data['trend']}\n"

        f"Structure: "
        f"{data['market_structure']}\n"

        f"Momentum: "
        f"{data['momentum']}\n\n"

        "🕯 CANDLE PSYCHOLOGY\n"

        f"{data['candle_psychology']}\n\n"

        "📍 SUPPORT / RESISTANCE\n"

        f"Support: "
        f"{data['support']}\n"

        f"Resistance: "
        f"{data['resistance']}\n\n"

        "💥 PRICE ACTION\n"

        f"Breakout: "
        f"{data['breakout']}\n"

        f"Trap: "
        f"{data['trap']}\n"

        f"Continuation: "
        f"{data['continuation']}\n"

        f"Exhaustion: "
        f"{data['exhaustion']}\n"

        f"Reversal: "
        f"{data['reversal']}\n\n"

        "🛡 QUALITY FILTER\n"

        f"Confirmation: "
        f"{data['confirmation']}\n"

        f"Risk: "
        f"{data['risk_level']}\n\n"

        f"📝 {data['reason']}\n\n"

        "⚠️ Analysis only. "
        "No guaranteed outcome."
    )


# =========================================================
# BUILD HISTORY CONTEXT
# =========================================================

def build_history_context(
    chat_id
) -> str:

    history = chat_history.get(
        chat_id
    )

    if not history:
        return (
            "No previous chart analysis "
            "is available for this chat."
        )

    lines = [
        "Previous analysis context "
        "(do not treat it as current market data):"
    ]

    for index, item in enumerate(
        history,
        start=1
    ):

        lines.append(
            f"{index}. {item}"
        )

    return "\n".join(lines)


# =========================================================
# SAVE HISTORY
# =========================================================

def save_history(
    chat_id,
    data
):

    summary = (
        f"Signal={data['next_candle']}; "
        f"UP={data['up_probability']}%; "
        f"DOWN={data['down_probability']}%; "
        f"Confidence={data['confidence']}%; "
        f"Trend={data['trend']}; "
        f"Structure={data['market_structure']}; "
        f"Momentum={data['momentum']}; "
        f"Support={data['support']}; "
        f"Resistance={data['resistance']}; "
        f"Confirmation={data['confirmation']}; "
        f"Risk={data['risk_level']}"
    )

    chat_history[
        chat_id
    ].append(summary)


# =========================================================
# ANALYZE CHART
# =========================================================

def analyze_chart(
    image_bytes: bytes,
    chat_id=None
) -> tuple[str, dict]:

    image_bytes = prepare_image(
        image_bytes
    )

    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")


    history_context = (
        build_history_context(
            chat_id
        )
        if chat_id is not None
        else "No previous context."
    )


    user_prompt = f"""
Analyze the current chart screenshot.

IMPORTANT:

The previous analysis context below is historical
context only. Do NOT assume that previous predictions
were correct.

Current screenshot is the primary source of truth.

{history_context}

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
                        )
                    }

                ]
            }

        ]
    )


    raw_result = (
        response.output_text
        .strip()
    )


    print(
        "RAW AI RESULT:",
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


    formatted = format_result(
        data
    )


    return formatted, data


# =========================================================
# /START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "🤖 QUOTEX AI ASSISTANT V3\n\n"

        "Send a clear trading chart screenshot.\n\n"

        "🧠 Analysis layers:\n"
        "• Candle Psychology\n"
        "• Market Structure\n"
        "• Momentum\n"
        "• Support / Resistance\n"
        "• Breakout / Fake Breakout\n"
        "• Trap Detection\n"
        "• Continuation\n"
        "• Exhaustion\n"
        "• Reversal\n"
        "• OTC Caution\n"
        "• Quality Filter\n\n"

        f"🎯 Minimum confidence: "
        f"{MIN_CONFIDENCE}%\n\n"

        "Weak setup → NO TRADE"
    )


# =========================================================
# /HELP
# =========================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "📖 QUOTEX AI ASSISTANT V3\n\n"

        "1. Open your chart.\n"
        "2. Make candles clearly visible.\n"
        "3. Send screenshot.\n"
        "4. Wait for analysis.\n\n"

        "The system checks:\n"
        "🕯 Candle psychology\n"
        "📊 Market structure\n"
        "📈 Momentum\n"
        "📍 Support / Resistance\n"
        "💥 Breakouts\n"
        "🪤 Traps\n"
        "🔄 Continuation / Reversal\n"
        "🥵 Exhaustion\n"
        "🟣 OTC caution\n"
        "🛡 Quality filter\n\n"

        "Use /clear to remove previous "
        "analysis context."
    )


# =========================================================
# /STATUS
# =========================================================

async def status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "🟢 QUOTEX AI ASSISTANT V3\n\n"

        f"Model: {OPENAI_MODEL}\n"

        f"Confidence threshold: "
        f"{MIN_CONFIDENCE}%\n"

        f"History: {MAX_HISTORY} analyses\n"

        "Mode: Quality-first\n"

        "Weak setup: NO TRADE\n"

        "Status: ONLINE"
    )


# =========================================================
# /CLEAR
# =========================================================

async def clear_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.effective_chat:

        return

    chat_id = update.effective_chat.id

    chat_history.pop(
        chat_id,
        None
    )

    await update.message.reply_text(

        "🧹 Previous analysis context cleared."
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
            "🔍 Reading chart...\n"
            "🧠 Checking market structure...\n"
            "🕯 Checking candle psychology...\n"
            "📍 Checking support/resistance..."
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


        result, data = analyze_chart(
            image_bytes,
            chat_id
        )


        await status_message.edit_text(
            result
        )


    except Exception as error:

        print(
            "PHOTO ANALYSIS ERROR:",
            repr(error)
        )


        await status_message.edit_text(

            "❌ Analysis failed.\n\n"

            "Please send a clear chart screenshot "
            "again.\n\n"

            "If the problem continues, check "
            "Render logs and API settings."
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

        "📊 Please send a trading chart screenshot.\n\n"

        "/help — instructions\n"
        "/status — bot status\n"
        "/clear — clear analysis context"
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
        "      QUOTEX AI ASSISTANT V3"
    )

    print(
        "=========================================="
    )

    print(
        f"Model: {OPENAI_MODEL}"
    )

    print(
        f"Minimum confidence: {MIN_CONFIDENCE}%"
    )

    print(
        f"History per chat: {MAX_HISTORY}"
    )


    # -----------------------------------------------------
    # RENDER WEB SERVER
    # -----------------------------------------------------

    server_thread = threading.Thread(

        target=run_web_server,

        daemon=True
    )

    server_thread.start()


    # -----------------------------------------------------
    # TELEGRAM APPLICATION
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
    # ERROR
    # -----------------------------------------------------

    telegram_app.add_error_handler(
        error_handler
    )


    print(
        "Telegram bot started."
    )

    print(
        "Quotex AI Assistant V3 is ready."
    )


    # -----------------------------------------------------
    # POLLING
    # -----------------------------------------------------

    telegram_app.run_polling(

        drop_pending_updates=True
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()






