import os
import logging
import httpx
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, filters, MessageHandler, InlineQueryHandler
from uuid import uuid4

from dotenv import load_dotenv
from classifier import load_model, classify
load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")
    

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)


async def caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text_caps = ' '.join(context.args).upper()
        await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)
    except:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please enter some text to caps.")


async def inline_caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query
    if not query:
        return
    results = [InlineQueryResultArticle(
        id=str(uuid4()),
        title="Caps",
        input_message_content=InputTextMessageContent(query.upper()),
    )]
    await context.bot.answer_inline_query(update.inline_query.id, results)


async def inline_predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query
    try:
        async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
            resp = await client.get(query, timeout=10, follow_redirects=True)
            resp.raise_for_status()
        letter, confidence = classify(resp.content)
        results = [InlineQueryResultArticle(
            id=str(uuid4()),
            title=f"Predicted: {letter} ({confidence:.1%})",
            description="Tap to send prediction",
            input_message_content=InputTextMessageContent(
                f"Predicted letter: *{letter}* (confidence: {confidence:.1%})",
                parse_mode="Markdown",
            ),
        )]
    except Exception as e:
        results = [InlineQueryResultArticle(
            id=str(uuid4()),
            title="Could not classify image",
            description=str(e)[:64],
            input_message_content=InputTextMessageContent("Failed to classify the image."),
        )]
    await context.bot.answer_inline_query(update.inline_query.id, results)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = bytes(await photo_file.download_as_bytearray())
    letter, confidence = classify(image_bytes)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Predicted letter: *{letter}* (confidence: {confidence:.1%})",
        parse_mode="Markdown"
    )


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")


if __name__ == '__main__':
    load_model()

    application = ApplicationBuilder().token(os.getenv("TELEGRAM_API_TOKEN")).build()

    unknown_handler = MessageHandler(filters.COMMAND, unknown)
    start_handler = CommandHandler('start', start)
    message_handler = MessageHandler((filters.TEXT & ~filters.COMMAND), echo)
    caps_handler = CommandHandler('caps', caps)
    inline_caps_handler = InlineQueryHandler(inline_caps, pattern=r"^(?!https?://)")
    inline_predict_handler = InlineQueryHandler(inline_predict, pattern=r"^https?://")
    photo_handler = MessageHandler(filters.PHOTO, handle_photo)

    application.add_handler(start_handler)
    application.add_handler(message_handler)
    application.add_handler(caps_handler)
    application.add_handler(inline_caps_handler)
    application.add_handler(inline_predict_handler)
    application.add_handler(photo_handler)

    application.add_handler(unknown_handler)  # add last, to let actual handlers trigger first

    application.run_polling()