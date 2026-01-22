"""
Configuration module for AI Portrait Telegram Bot
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Google AI Configuration (for Gemini Vision and Imagen)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_VISION_MODEL = os.getenv('GOOGLE_VISION_MODEL', 'gemini-1.5-pro')
GOOGLE_IMAGE_MODEL = os.getenv('GOOGLE_IMAGE_MODEL', 'imagen-3.0-generate-001')

# Validate required configuration
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env file")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env file")

# Image settings
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
SUPPORTED_IMAGE_FORMATS = ['image/jpeg', 'image/png', 'image/jpg']

# Bot messages (Russian)
MESSAGES = {
    'start': (
        '👋 Добро пожаловать в AI Portrait Bot!\n\n'
        'Я помогу превратить ваше селфи в художественный портрет.\n\n'
        '🎨 Как это работает:\n'
        '1. Отправьте мне картинку-образец (стиль, который хотите применить)\n'
        '2. Затем отправьте фотографию человека\n'
        '3. Получите уникальный портрет!\n\n'
        'Отправьте картинку-образец для начала работы.'
    ),
    'waiting_reference': '📸 Пожалуйста, отправьте картинку-образец (стиль для вашего портрета).',
    'reference_received': (
        '✅ Образец получен!\n\n'
        'Теперь отправьте фотографию человека, которого нужно изобразить в этом стиле.'
    ),
    'person_received': '⚙️ Отлично! Начинаю работу над портретом...\n\nЭто может занять 30-60 секунд.',
    'processing': '🎨 Создаю ваш шедевр...',
    'success': '✨ Готово! Вот ваш AI-портрет.\n\nХотите создать еще один? Отправьте новую картинку-образец.',
    'error': '❌ Произошла ошибка: {error}\n\nПопробуйте еще раз или отправьте /start для начала.',
    'invalid_image': '⚠️ Пожалуйста, отправьте корректное изображение (JPEG или PNG).',
    'help': (
        '🤖 AI Portrait Bot - Справка\n\n'
        'Команды:\n'
        '/start - Начать работу с ботом\n'
        '/help - Показать эту справку\n'
        '/cancel - Отменить текущую операцию\n\n'
        'Процесс создания портрета:\n'
        '1. Отправьте картинку-образец\n'
        '2. Отправьте фото человека\n'
        '3. Получите результат\n'
    ),
    'cancelled': '🚫 Операция отменена. Отправьте /start для начала работы.'
}
