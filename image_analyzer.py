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
        self.text_model = config.GOOGLE_TEXT_MODEL  # For text analysis stages 1-2

    def detect_aspect_ratio(self, image: Image.Image) -> str:
        """
        Detect aspect ratio from image dimensions (like working app)

        Args:
            image: PIL Image object

        Returns:
            Aspect ratio string (1:1, 16:9, 9:16, 4:3, 3:4)
        """
        width, height = image.size
        ratio = width / height

        # Determine closest supported aspect ratio for Gemini API
        if ratio < 0.65:
            return '9:16'
        elif ratio < 0.85:
            return '3:4'
        elif ratio < 1.2:
            return '1:1'
        elif ratio < 1.5:
            return '4:3'
        else:
            return '16:9'

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

        # Detect aspect ratio from image dimensions (like working app)
        aspect_ratio = self.detect_aspect_ratio(image)
        logger.info(f"Detected aspect ratio from reference: {aspect_ratio}")

        # Детальный промпт для анализа медиума и стиля (как в рабочем приложении)
        prompt = """Perform a comprehensive forensic analysis of this image:

STAGE 1: MEDIUM CLASSIFICATION
Identify the exact medium: Is it a high-fidelity Photograph, an Oil/Acrylic Painting, a Watercolor, a Pencil/Charcoal Sketch, an Etching, or Digital Art?

STAGE 2: MEDIUM-SPECIFIC TECHNICAL SPECIFICATION
- IF PHOTOGRAPHY: Describe focal length, aperture (depth of field), film stock grain, lighting setup (e.g., Rembrandt, butterfly), and physical print texture (glossy, matte, silver gelatin).
- IF PAINTING/WATERCOLOR: Describe brushwork scale, impasto thickness, canvas weave density, paper "tooth" (cold-press texture), pigment bleeding, and water stains.
- IF SKETCH: Describe graphite/charcoal grit, smudge marks, eraser ghosts, hatching density, and paper fibers.
- IF DIGITAL: Identify the specific software aesthetic (brush engines used).

STAGE 3: COMPOSITION & CHARACTER (CRITICAL - DESCRIBE IN DETAIL)
1. POSE & PLACEMENT: Exact posture, body position, and placement in frame (centered, off-center, rule of thirds, etc.)
2. CLOTHING & ACCESSORIES: Describe every detail - fabric type, color, style, era, buttons, jewelry, hat, everything visible
3. ENVIRONMENT & BACKGROUND: Complete description - indoor/outdoor, objects, scenery, depth, foreground/background elements
4. ATMOSPHERE: Mood, time of day, weather, emotional tone, color temperature (warm/cool)
5. LIGHTING: Direction, intensity, shadows, highlights, color of light
6. COLORS: Dominant colors, color palette, saturation, contrast
7. CAMERA ANGLE: Eye level, low angle, high angle, perspective
8. ASPECT RATIO: Exact ratio (1:1, 16:9, 9:16, 4:3, 3:4, etc.)

Output a complete specification that includes ALL these details so the image can be recreated exactly.

IMPORTANT: Return response in JSON format:
{
  "medium_specification": "detailed medium and technique description",
  "clothing": "complete clothing and accessories description",
  "environment": "complete background and environment description",
  "atmosphere": "mood, lighting, time of day, emotional tone",
  "colors": "color palette and scheme",
  "pose": "exact body position and posture",
  "composition": "framing, placement, camera angle",
  "aspect_ratio": "aspect ratio like 1:1, 16:9, etc."
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

                    # Generate content with new SDK using TEXT model (not image model)
                    response = self.client.models.generate_content(
                        model=self.text_model,
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

            # Override aspect_ratio with detected value from image dimensions
            # (Gemini may return wrong value, we calculate from actual image)
            result['aspect_ratio'] = aspect_ratio

            logger.info("Reference image analysis completed")
            logger.info(f"Using aspect ratio from reference dimensions: {aspect_ratio}")
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

                    # Generate content with new SDK using TEXT model (not image model)
                    response = self.client.models.generate_content(
                        model=self.text_model,
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

    def refine_with_person(
        self,
        reference_data: Dict[str, str],
        person_image_path: str
    ) -> str:
        """
        Refine reference specification with person's photo to create final generation prompt

        Args:
            reference_data: Complete reference analysis (medium, clothing, environment, etc.)
            person_image_path: Path to person's photo

        Returns:
            Final generation prompt harmonizing identity with reference scene
        """
        logger.info("Refining prompt with person photo...")

        # Compress person image
        image = self.compress_image(person_image_path, max_size=1024)

        # Comprehensive prompt that includes ALL elements from reference
        medium_spec = reference_data.get('medium_specification', '')
        clothing = reference_data.get('clothing', '')
        environment = reference_data.get('environment', '')
        atmosphere = reference_data.get('atmosphere', '')
        colors = reference_data.get('colors', '')
        pose = reference_data.get('pose', '')
        composition = reference_data.get('composition', '')

        prompt = f"""REFERENCE IMAGE COMPLETE SPECIFICATION:

MEDIUM & TECHNIQUE: {medium_spec}

CLOTHING & OUTFIT: {clothing}

ENVIRONMENT & BACKGROUND: {environment}

ATMOSPHERE: {atmosphere}

COLORS: {colors}

POSE & BODY POSITION: {pose}

COMPOSITION: {composition}

---

TASK: Create a portrait of the person in this photo, but place them INTO the scene described above.

CRITICAL INSTRUCTIONS:
1. FACE IDENTITY: Use the EXACT face from this photo - this person must be 100% recognizable (facial features, proportions, age, ethnicity)
2. CLOTHING: Use the CLOTHING from the reference specification above (adapt for gender if needed, but keep style/era/fabric)
3. ENVIRONMENT: Use the EXACT BACKGROUND and environment from specification (same setting, objects, scenery)
4. POSE: Use the EXACT BODY POSITION and posture from specification
5. ATMOSPHERE: Match the EXACT mood, lighting, time of day from specification
6. COLORS: Use the EXACT color palette from specification
7. COMPOSITION: Match the EXACT framing and camera angle from specification
8. MEDIUM INTEGRITY: If photo - realistic textures. If painting - visible brushstrokes. If sketch - pencil marks.
9. NO SMOOTHING: Demand raw texture matching the medium (film grain / canvas texture / paper fibers)

Think of this as: Take this person's face and place them into that exact scene with that exact clothing, pose, lighting, and atmosphere.

Output a single comprehensive generation prompt that combines everything."""

        try:
            # Convert image to base64
            image_b64 = self._image_to_base64(image)

            # Retry logic
            max_retries = 3
            retry_delays = [2, 4, 8]
            response = None

            for attempt in range(max_retries):
                try:
                    logger.info(f"Refine prompt attempt {attempt + 1}/{max_retries}")

                    response = self.client.models.generate_content(
                        model=self.text_model,
                        contents={
                            'parts': [
                                {'inline_data': {'mime_type': 'image/jpeg', 'data': image_b64}},
                                {'text': prompt}
                            ]
                        }
                    )

                    logger.info("Prompt refinement successful")
                    break

                except Exception as api_error:
                    error_msg = str(api_error)
                    is_temporary = ('503' in error_msg or 'UNAVAILABLE' in error_msg or
                                  '429' in error_msg or 'overloaded' in error_msg.lower() or
                                  'rate limit' in error_msg.lower())

                    if is_temporary and attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        logger.warning(f"Temporary API error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"API call failed: {error_msg}")
                        raise

            if response is None:
                logger.error("Failed to get response after all retry attempts")
                raise Exception("Failed to refine prompt with person")

            refined_prompt = response.text.strip()
            logger.info("Prompt refinement completed")
            return refined_prompt

        except Exception as e:
            logger.error(f"Error refining prompt with person: {e}")
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

        # Создаем финальный промпт: используем full_prompt для детального стиля
        # full_prompt содержит полное описание стиля от Gemini
        style_description = reference_data.get('full_prompt',
            f"{reference_data.get('image_type', 'portrait')} - {reference_data.get('technical_details', 'professional style')}. "
            f"{reference_data.get('color_palette', 'natural colors')}. {reference_data.get('lighting', 'natural lighting')}. "
            f"{reference_data.get('background', 'neutral background')}. {reference_data.get('mood', 'neutral mood')}.")

        final_prompt = f"""STYLE: {style_description}

COMPOSITION: {reference_data.get('shot_scale', 'portrait')} shot. {reference_data.get('camera_angle', 'eye level')}. {reference_data.get('subject_placement', 'centered')}.

POSE: {reference_data.get('subject_pose', 'natural pose')}. {reference_data.get('subject_dynamics', 'static')}. {reference_data.get('facial_expression', 'neutral expression')}. {reference_data.get('clothing', 'casual attire')}{gender_swap_note}.

PERSON: {person_data.get('gender', 'adult')}, ~{person_data.get('age', 'adult')} years. Hair: {person_data.get('hair_color', 'natural')} {person_data.get('hair_length', 'medium')} {person_data.get('hair_style', 'natural style')}."""

        aspect_ratio = reference_data.get('aspect_ratio', '1:1')

        logger.info("Prompt merging completed")
        return final_prompt, aspect_ratio
