"""
Image Analysis Module using Google Gemini Vision API
Analyzes reference images and person photos to create detailed prompts
"""
import logging
import json
import io
import base64
import time
from typing import Dict, Tuple
from PIL import Image
from google import genai
import config

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Analyzes images using Google Gemini Vision API"""

    def __init__(self):
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self.model_name = config.GOOGLE_VISION_MODEL

    def compress_image(self, image_path: str, max_size: int = 1024) -> Image.Image:
        """
        Compress image to reduce API processing time

        Args:
            image_path: Path to the image file
            max_size: Maximum dimension (width or height) in pixels

        Returns:
            Compressed PIL Image
        """
        image = Image.open(image_path)

        # Get current size
        width, height = image.size

        # Calculate new size maintaining aspect ratio
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            logger.info(f"Compressing image from {width}x{height} to {new_width}x{new_height}")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to RGB if needed (remove alpha channel)
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = rgb_image

        return image

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string for new SDK

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def analyze_reference_image(self, image_path: str) -> Dict[str, str]:
        """
        Analyze reference image to extract detailed style and composition

        Args:
            image_path: Path to the reference image

        Returns:
            Dictionary with style description, composition, and technical details
        """
        logger.info(f"Analyzing reference image: {image_path}")

        # Загружаем и сжимаем изображение для ускорения обработки
        image = self.compress_image(image_path, max_size=1024)

        # Детальный промпт для анализа образца согласно алгоритму
        prompt = """Проанализируй это изображение максимально подробно для создания AI-портрета.

ВАЖНО: Опиши изображение так, чтобы можно было точно воссоздать стиль и композицию.

1. ОБЩАЯ СТИЛИСТИКА (КРИТИЧЕСКИ ВАЖНО):
   - Определи тип изображения (фотография/векторная графика/рисунок/3D-рендер/другое)
   - Если ВЕКТОРНАЯ ГРАФИКА: опиши детально:
     * Стиль линий: чистые контуры, толщина линий, черные обводки, цветные контуры
     * Заливка: плоские цвета, градиенты, без текстур, с паттернами
     * Уровень детализации: минималистичный, детальный, фотореалистичный вектор
     * Характерные черты: flat design, линейная графика, силуэты
   - Если фото: укажи тип камеры, объектив, выдержку, диафрагму, ISO (если можно определить по виду)
   - Если рисунок: тип краски/карандаша, текстуру, технику, характерные штрихи
   - Если иллюстрация: стиль (реализм/мультяшный/комикс/аниме/другое)

2. ФОРМАТ И КОМПОЗИЦИЯ (КРИТИЧЕСКИ ВАЖНО):
   - Соотношение сторон (1:1, 16:9, 9:16, 4:3, 3:4, другое)
   - Ориентация (портретная/альбомная/квадратная)
   - Расположение субъекта в кадре (точно!):
     * По горизонтали: центр, левая треть, правая треть, золотое сечение слева/справа
     * По вертикали: центр, верхняя треть, нижняя треть, золотое сечение сверху/снизу
     * Пример: "объект на линии золотого сечения слева, занимает 2/3 площади слева"
   - Крупность плана (детально!):
     * Портрет: от подбородка до макушки, от шеи до макушки
     * Поясной: от талии и выше
     * По колено, по бедра
     * Ростовой: в полный рост
     * Деталь: только лицо крупным планом
   - Угол камеры (точно!):
     * Высота: прямо на уровне глаз / сверху (указать ~30°/~45°/~60°) / снизу (указать угол)
     * Горизонталь: фронтально (прямо 0°) / три четверти (~30-45°) / профиль (90°) / со спины
     * Перспектива: плоская фронтальная / угловая с глубиной / диагональная композиция

3. СУБЪЕКТ (ЕСЛИ ЕСТЬ ЧЕЛОВЕК):
   - Поза и положение тела (МАКСИМАЛЬНО ДЕТАЛЬНО):
     * Точное положение: сидит/стоит/лежит/в движении
     * Положение рук, ног, головы
     * Поворот корпуса, наклон головы
   - Динамика и движение:
     * Статичная поза или в движении
     * Волосы развеваются / статичные
     * Одежда в движении / спокойная
   - Выражение лица и эмоции
   - Одежда и аксессуары (детально!)
   - Прическа и ее стиль
   - Освещение на лице

4. ОКРУЖЕНИЕ И ДЕТАЛИ:
   - Фон (детальное описание)
   - Окружающие предметы и элементы
   - Освещение (мягкое/жесткое, направление света, цветовая температура)
   - Цветовая палитра (теплая/холодная, насыщенность, контраст)

5. НАСТРОЕНИЕ И АТМОСФЕРА:
   - Общее настроение изображения
   - Эмоциональный посыл
   - Художественные эффекты

Ответ дай СТРОГО в формате JSON (без дополнительного текста, только JSON):
{
  "image_type": "тип изображения",
  "technical_details": "технические детали стиля",
  "aspect_ratio": "соотношение сторон",
  "composition": "описание композиции и размещения субъекта",
  "subject_placement": "точное расположение в кадре (трети, золотое сечение)",
  "shot_scale": "крупность плана (портрет, поясной, ростовой и т.д.)",
  "camera_angle": "угол камеры (высота и поворот)",
  "subject_pose": "детальное описание позы тела",
  "subject_dynamics": "динамика движения (волосы, одежда, движение)",
  "facial_expression": "выражение лица и эмоции",
  "clothing": "одежда и аксессуары",
  "hairstyle": "прическа",
  "background": "описание фона",
  "lighting": "освещение",
  "color_palette": "цветовая палитра",
  "mood": "настроение",
  "full_prompt": "полный промпт для генерации в стиле этого изображения"
}"""

        try:
            # Convert image to base64 for new SDK
            image_b64 = self._image_to_base64(image)

            # Retry logic for handling temporary API overload (503, 429)
            max_retries = 3
            retry_delays = [2, 4, 8]  # Exponential backoff: 2s, 4s, 8s
            response = None

            for attempt in range(max_retries):
                try:
                    logger.info(f"Reference analysis attempt {attempt + 1}/{max_retries}")

                    # Generate content with new SDK
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents={
                            'parts': [
                                {'text': prompt},
                                {'inline_data': {'mime_type': 'image/jpeg', 'data': image_b64}}
                            ]
                        }
                    )

                    # If successful, break out of retry loop
                    logger.info("Reference analysis successful")
                    break

                except Exception as api_error:
                    error_msg = str(api_error)

                    # Check if it's a temporary error (503 overload, 429 rate limit)
                    is_temporary = ('503' in error_msg or 'UNAVAILABLE' in error_msg or
                                  '429' in error_msg or 'overloaded' in error_msg.lower() or
                                  'rate limit' in error_msg.lower())

                    if is_temporary and attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        logger.warning(f"Temporary API error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        # Either not temporary error, or final attempt failed
                        logger.error(f"API call failed: {error_msg}")
                        raise

            # Check if we got a response after retries
            if response is None:
                logger.error("Failed to get response after all retry attempts")
                raise Exception("Failed to analyze reference image after retries")

            # Очищаем ответ от markdown форматирования
            result_text = response.text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            result = json.loads(result_text)
            logger.info("Reference image analysis completed")
            return result

        except Exception as e:
            logger.error(f"Error analyzing reference image: {e}")
            raise

    def analyze_person_image(self, image_path: str) -> Dict[str, str]:
        """
        Analyze person's photo to extract facial features and characteristics

        Args:
            image_path: Path to the person's photo

        Returns:
            Dictionary with person's characteristics
        """
        logger.info(f"Analyzing person image: {image_path}")

        # Загружаем и сжимаем изображение для ускорения обработки
        image = self.compress_image(image_path, max_size=1024)

        # Упрощенный анализ человека - фокус на базовых характеристиках
        # Сходство лица будет определяться самим изображением, а не текстовым описанием
        prompt = """Проанализируй базовые характеристики этого человека для AI-портрета.

ВАЖНО: Опиши только основные атрибуты. Черты лица НЕ описывай - они будут взяты с фотографии.

1. БАЗОВАЯ ИНФОРМАЦИЯ:
   - Пол (мужчина/женщина)
   - Примерный возраст

2. ВОЛОСЫ (ПОДРОБНО):
   - Цвет волос (точный оттенок: блонд, рыжий, брюнет, шатен, седой и т.д.)
   - Длина волос (очень короткие/короткие/до плеч/длинные/очень длинные)
   - Стиль прически (прямые/волнистые/кудрявые, распущены/собраны/зачесаны назад/на бок и т.д.)

Ответ дай СТРОГО в формате JSON (без дополнительного текста, только JSON):
{
  "gender": "пол",
  "age": "возраст",
  "hair_color": "цвет волос",
  "hair_length": "длина волос",
  "hair_style": "стиль прически"
}"""

        try:
            # Convert image to base64 for new SDK
            image_b64 = self._image_to_base64(image)

            # Retry logic for handling temporary API overload (503, 429)
            max_retries = 3
            retry_delays = [2, 4, 8]  # Exponential backoff: 2s, 4s, 8s
            response = None

            for attempt in range(max_retries):
                try:
                    logger.info(f"Person analysis attempt {attempt + 1}/{max_retries}")

                    # Generate content with new SDK
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents={
                            'parts': [
                                {'text': prompt},
                                {'inline_data': {'mime_type': 'image/jpeg', 'data': image_b64}}
                            ]
                        }
                    )

                    # If successful, break out of retry loop
                    logger.info("Person analysis successful")
                    break

                except Exception as api_error:
                    error_msg = str(api_error)

                    # Check if it's a temporary error (503 overload, 429 rate limit)
                    is_temporary = ('503' in error_msg or 'UNAVAILABLE' in error_msg or
                                  '429' in error_msg or 'overloaded' in error_msg.lower() or
                                  'rate limit' in error_msg.lower())

                    if is_temporary and attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        logger.warning(f"Temporary API error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        # Either not temporary error, or final attempt failed
                        logger.error(f"API call failed: {error_msg}")
                        raise

            # Check if we got a response after retries
            if response is None:
                logger.error("Failed to get response after all retry attempts")
                raise Exception("Failed to analyze person image after retries")

            # Очищаем ответ от markdown форматирования
            result_text = response.text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            result = json.loads(result_text)
            logger.info("Person image analysis completed")
            return result

        except Exception as e:
            logger.error(f"Error analyzing person image: {e}")
            raise

    def merge_prompts(
        self,
        reference_data: Dict[str, str],
        person_data: Dict[str, str]
    ) -> Tuple[str, str]:
        """
        Merge reference style with person characteristics

        Args:
            reference_data: Analysis results from reference image
            person_data: Analysis results from person's photo

        Returns:
            Tuple of (final_prompt, aspect_ratio)
        """
        logger.info("Merging prompts for final generation")

        # Проверка на изменение пола и корректировка одежды
        reference_has_person = 'subject_pose' in reference_data and reference_data['subject_pose']
        gender_swap_note = ""

        if reference_has_person:
            # Если пол отличается, добавляем заметку о замене одежды
            person_gender = person_data.get('gender', '').lower()
            if person_gender:
                gender_swap_note = f"\n\nIMPORTANT: The subject is {person_gender}. "
                if 'female' in person_gender or 'женщина' in person_gender:
                    gender_swap_note += "If the reference has masculine clothing, replace it with appropriate feminine clothing while maintaining the style."
                else:
                    gender_swap_note += "If the reference has feminine clothing, replace it with appropriate masculine clothing while maintaining the style."

        # Создаем финальный промпт: короткий, но информативный, с правильными приоритетами
        final_prompt = f"""STYLE: {reference_data.get('image_type', 'portrait')} - {reference_data.get('technical_details', 'professional style')}. {reference_data.get('color_palette', 'natural colors')}. {reference_data.get('lighting', 'natural lighting')}. {reference_data.get('background', 'neutral background')}. {reference_data.get('mood', 'neutral mood')}.

COMPOSITION: {reference_data.get('shot_scale', 'portrait')} shot. {reference_data.get('camera_angle', 'eye level')}. {reference_data.get('subject_placement', 'centered')}.

POSE: {reference_data.get('subject_pose', 'natural pose')}. {reference_data.get('subject_dynamics', 'static')}. {reference_data.get('facial_expression', 'neutral expression')}. {reference_data.get('clothing', 'casual attire')}{gender_swap_note}.

PERSON: {person_data.get('gender', 'adult')}, ~{person_data.get('age', 'adult')} years. Hair: {person_data.get('hair_color', 'natural')} {person_data.get('hair_length', 'medium')} {person_data.get('hair_style', 'natural style')}."""

        aspect_ratio = reference_data.get('aspect_ratio', '1:1')

        logger.info("Prompt merging completed")
        return final_prompt, aspect_ratio
