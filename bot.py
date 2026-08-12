import os
import base64
import threading

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

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable is missing.")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is missing.")

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================================================
# RENDER HEALTH SERVER
# =========================================================

app = Flask(__name__)


@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "bot": "FAYA AI Trading Signal Assistant"
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


def run_web_server():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)


# =========================================================
# AI SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are Quotex AI Assistant.

Analyze the provided trading chart image carefully.

Analyze:
- Short-term trend
- Momentum strength
- Continuation or exhaustion
- Possible traps
- Support
- Resistance
- Confirmation

SIGNAL RULE:

1. Estimate UP probability and DOWN probability.
2. UP + DOWN MUST equal exactly 100%.
3. CONFIDENCE MUST equal the higher of UP and DOWN.
4. If UP is 60% or higher:
   NEXT CANDLE = UP
5. If DOWN is 60% or higher:
   NEXT CANDLE = DOWN
6. Only if BOTH UP and DOWN are below 60%:
   NEXT CANDLE = NO TRADE
7. Never claim certainty or guaranteed accuracy.
8. Never invent unseen market data.

IMPORTANT EXAMPLES:

If UP = 61% and DOWN = 39%:
NEXT CANDLE = UP

If UP = 59% and DOWN = 41%:
NEXT CANDLE = NO TRADE

If UP = 30% and DOWN = 70%:
NEXT CANDLE = DOWN

OUTPUT ONLY THIS EXACT FORMAT:

📊 QUOTEX AI ANALYSIS

NEXT CANDLE: UP / DOWN / NO TRADE
UP: 0%
DOWN: 0%
CONFIDENCE: 0%

TREND: Bullish / Bearish / Unclear
MOMENTUM: Strong / Moderate / Weak
SUPPORT: Confirmed / Weak / Unclear
RESISTANCE: Confirmed / Weak / Unclear
CONFIRMATION: Good / Weak / Unclear
"""


# =========================================================
# ANALYZE IMAGE
# =========================================================

def analyze_chart(image_bytes: bytes) -> str:

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model="gpt-5-mini",
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
                        "text": "Analyze this trading chart."
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    result = response.output_text.strip()

    return result


# =========================================================
# START COMMAND
# =========================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "🤖 QUOTEX AI ASSISTANT\n"
        "Send me a trading chart screenshot.\n"
        "I will analyze it and return:\n\n"
        "NEXT CANDLE\n"
        "UP probability\n"
        "DOWN probability\n"
        "CONFIDENCE\n"
        "TREND\n"
        "MOMENTUM\n"
        "SUPPORT\n"
        "RESISTANCE\n"
        "CONFIRMATION"
    )


# =========================================================
# PHOTO HANDLER
# =========================================================

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not update.message or not update.message.photo:
        return

    status_message = await update.message.reply_text(
        "🔍 Analyzing chart..."
    )

    try:

        photo = update.message.photo[-1]

        telegram_file = await photo.get_file()

        image_bytes = bytes(
            await telegram_file.download_as_bytearray()
        )

        result = analyze_chart(image_bytes)

        await status_message.edit_text(result)

    except Exception as e:

        print("ERROR:", repr(e))

        await status_message.edit_text(
            "❌ Analysis error.\n"
            "Please try sending the chart again."
        )


# =========================================================
# TEXT HANDLER
# =========================================================

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not update.message:
        return

    await update.message.reply_text(
        "📊 Please send a trading chart screenshot."
    )


# =========================================================
# ERROR HANDLER
# =========================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):

    print("Telegram error:", repr(context.error))


# =========================================================
# MAIN
# =========================================================

def main():

    print("===================================")
    print("QUOTEX AI ASSISTANT")
    print("Starting...")
    print("===================================")

    # Start Render health server
    server_thread = threading.Thread(
        target=run_web_server,
        daemon=True
    )

    server_thread.start()

    # Telegram application
    telegram_app = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    telegram_app.add_handler(
        CommandHandler("start", start)
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

    telegram_app.add_error_handler(error_handler)

    print("Telegram bot started.")
    print("Quotex AI Assistant is ready.")

    telegram_app.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
