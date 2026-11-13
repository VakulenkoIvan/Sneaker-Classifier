import os
import io
import telebot
from dotenv import load_dotenv
from predict import SneakerClassifier

load_dotenv()  # загружает переменные из файла .env в окружение
TOKEN = os.getenv("BOT_TOKEN")

bot = telebot.TeleBot(TOKEN)

classifier = SneakerClassifier()  # загружаем нейросеть 1 раз при запуске


@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "Привет! Кидай фотку кроссовок, а я попробую угадать модель.")


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    # берём фото в максимальном качестве (последнее в списке)
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # передаём байты напрямую в наш классификатор без сохранения на диск
    predictions = classifier.predict(io.BytesIO(downloaded_file))

    lines = ["**Вот что я думаю:**\n"]
    for rank, (name, conf) in enumerate(predictions, 1):
        lines.append(f"{rank}. {name} — **{conf}%**")

    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown")


if __name__ == "__main__":
    print("Бот запущен...")
    bot.infinity_polling()