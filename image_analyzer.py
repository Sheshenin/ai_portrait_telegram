"""
Image Analysis Module using Google Gemini Vision API
Analyzes reference images and person photos to create detailed prompts
"""
import logging
import json
import io
from typing import Dict, Tuple
from PIL import Image
import google.generativeai as genai
import config

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Analyzes images using Google Gemini Vision API"""

    def __init__(self):
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(config.GOOGLE_VISION_MODEL)

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

1. ОБЩАЯ СТИЛИСТИКА:
   - Определи тип изображения (фотография/иллюстрация/рисунок/3D-рендер/другое)
   - Если фото: укажи тип камеры, объектив, выдержку, диафрагму, ISO (если можно определить по виду)
   - Если рисунок: тип краски/карандаша, текстуру, технику, характерные штрихи
   - Если иллюстрация: стиль (реализм/мультяшный/комикс/аниме/другое)

2. ФОРМАТ И КОМПОЗИЦИЯ:
   - Соотношение сторон (1:1, 16:9, 9:16, 4:3, 3:4, другое)
   - Ориентация (портретная/альбомная/квадратная)
   - Расположение главного субъекта (центр/смещение влево-вправо/верх-низ)
   - Угол съемки/изображения (прямо/сверху/снизу/сбоку)

3. СУБЪЕКТ (ЕСЛИ ЕСТЬ ЧЕЛОВЕК):
   - Поза и положение тела
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
  "composition": "описание композиции",
  "subject_pose": "поза субъекта",
  "clothing": "одежда и аксессуары",
  "hairstyle": "прическа",
  "background": "описание фона",
  "lighting": "освещение",
  "color_palette": "цветовая палитра",
  "mood": "настроение",
  "full_prompt": "полный промпт для генерации в стиле этого изображения"
}"""

        try:
            response = self.model.generate_content(
                [prompt, image],
                request_options={"timeout": 120}
            )

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
            response = self.model.generate_content(
                [prompt, image],
                request_options={"timeout": 120}
            )

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

        # Создаем финальный промпт с фокусом на стиль, а лицо берем из изображения
        final_prompt = f"""Create a portrait image with the following specifications:

STYLE AND TECHNIQUE:
- Image type: {reference_data.get('image_type', 'portrait')}
- Technical style: {reference_data.get('technical_details', 'professional portrait')}
- Color palette: {reference_data.get('color_palette', 'natural colors')}
- Lighting: {reference_data.get('lighting', 'natural lighting')}
- Mood: {reference_data.get('mood', 'neutral')}

COMPOSITION:
- Aspect ratio: {reference_data.get('aspect_ratio', '1:1')}
- Composition: {reference_data.get('composition', 'centered portrait')}
- Background: {reference_data.get('background', 'neutral background')}

SUBJECT - USE FACE FROM PROVIDED PHOTOGRAPH:
- Gender: {person_data.get('gender', 'adult')}
- Age: approximately {person_data.get('age', 'adult')}
- Hair color: {person_data.get('hair_color', 'natural hair color')}
- Hair length: {person_data.get('hair_length', 'medium length')}
- Hair style: {person_data.get('hair_style', 'natural style')}
- FACE: Use the exact face from the provided photograph (attached image)

POSE AND CLOTHING:
- Pose: {reference_data.get('subject_pose', 'natural pose')}
- Clothing: {reference_data.get('clothing', 'casual attire')}{gender_swap_note}

CRITICAL REQUIREMENTS:
1. USE THE EXACT FACE from the attached photograph - this is the #1 priority
2. The person's face MUST be 100% recognizable from the photograph
3. Apply ONLY the artistic style, composition, and pose from the reference description
4. Do NOT modify facial features - transfer the face as-is into the artistic style
5. The final image should look like: the person from the photo rendered in the reference style
"""

        aspect_ratio = reference_data.get('aspect_ratio', '1:1')

        logger.info("Prompt merging completed")
        return final_prompt, aspect_ratio
