import cv2
import telebot
from google.cloud import vision
from google.protobuf.json_format import MessageToJson
import io
import numpy as np

# Замените <YOUR_API_KEY> на ваш ключ API Google Cloud Vision
API_KEY = 'api key google cloud vosion'


# Создайте экземпляр клиента Google Cloud Vision
client = vision.ImageAnnotatorClient.from_service_account_json('credentials.json')

# Создайте экземпляр бота Telegram
bot = telebot.TeleBot('API key tg')

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hi! Send me an image for text recognition.")

# Обработчик получения изображения
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    # Получите идентификатор файла изображения
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    file_path = file_info.file_path
    
    # Скачайте изображение
    downloaded_file = bot.download_file(file_path)
    
    # Прочитайте изображение с помощью OpenCV
    image = cv2.imdecode(np.frombuffer(downloaded_file, np.uint8), -1)
    
    # Создайте объект Image для отправки в Google Cloud Vision
    success, encoded_image = cv2.imencode('.jpg', image)
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    
    # Выполните распознавание текста с помощью Google Cloud Vision API
    response = client.text_detection(image=image)
    
    # Преобразуйте ответ в формат JSON
    response_json = MessageToJson(response)
    
    # Извлеките распознанный текст из JSON
    extracted_text = response_json['textAnnotations'][0]['description'] if 'textAnnotations' in response_json else ''
    
    if extracted_text:
        bot.reply_to(message, extracted_text)
    else:
        bot.reply_to(message, "The text on the image could not be recognized.")

# Запустите бота
bot.polling()