import cv2
import telebot
from google.cloud import vision
from google.protobuf.json_format import MessageToJson
import io
import numpy as np

# �������� <YOUR_API_KEY> �� ��� ���� API Google Cloud Vision
API_KEY = 'api key google'

# �������� ��������� ������� Google Cloud Vision
client = vision.ImageAnnotatorClient.from_service_account_json('credentials.json')

# �������� ��������� ���� Telegram
bot = telebot.TeleBot('API key tg')

# ���������� ������� /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "1")

# ���������� ��������� �����������
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    # �������� ������������� ����� �����������
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    file_path = file_info.file_path
    
    # �������� �����������
    downloaded_file = bot.download_file(file_path)
    
    # ���������� ����������� � ������� OpenCV
    image = cv2.imdecode(np.frombuffer(downloaded_file, np.uint8), -1)
    
    # �������� ������ Image ��� �������� � Google Cloud Vision
    success, encoded_image = cv2.imencode('.jpg', image)
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    
    # ��������� ������������� ������ � ������� Google Cloud Vision API
    response = client.text_detection(image=image)
    
    # ������������ ����� � ������ JSON
    response_json = MessageToJson(response)
    
    # ��������� ������������ ����� �� JSON
    extracted_text = response_json['textAnnotations'][0]['description'] if 'textAnnotations' in response_json else ''
    
    if extracted_text:
        bot.reply_to(message, extracted_text)
    else:
        bot.reply_to(message, "The text on the image could not be recognized.")

# ��������� ����
bot.polling()