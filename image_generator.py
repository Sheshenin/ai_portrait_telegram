"""
Image Generation Module using Google Imagen
Generates AI portraits based on detailed prompts
"""
import logging
import base64
from typing import Optional, Tuple
import google.generativeai as genai
import config

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generates images using Google Imagen"""

    def __init__(self):
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.model_name = config.GOOGLE_IMAGE_MODEL

    def _map_aspect_ratio_to_ratio(self, aspect_ratio: str) -> str:
        """
        Map aspect ratio string to Imagen format

        Args:
            aspect_ratio: Aspect ratio string (e.g., "1:1", "16:9")

        Returns:
            Ratio string for Imagen (e.g., "1:1", "16:9", "9:16", "4:3", "3:4")
        """
        # Нормализуем формат
        ratio = aspect_ratio.lower().strip()

        # Маппинг распространенных форматов
        if ratio in ['1:1', '1/1', 'square', 'квадрат']:
            return '1:1'
        elif ratio in ['16:9', '16/9', 'landscape', 'альбом']:
            return '16:9'
        elif ratio in ['9:16', '9/16', 'portrait', 'портрет']:
            return '9:16'
        elif ratio in ['4:3', '4/3']:
            return '4:3'
        elif ratio in ['3:4', '3/4']:
            return '3:4'
        else:
            # По умолчанию квадрат
            logger.warning(f"Unknown aspect ratio: {ratio}, defaulting to 1:1")
            return '1:1'

    def generate_portrait(
        self,
        prompt: str,
        aspect_ratio: str = '1:1',
        number_of_images: int = 1
    ) -> Optional[bytes]:
        """
        Generate AI portrait using Google Imagen

        Args:
            prompt: Detailed prompt for image generation
            aspect_ratio: Desired aspect ratio (1:1, 16:9, 9:16, etc.)
            number_of_images: Number of images to generate (default 1)

        Returns:
            Image data as bytes, or None if failed
        """
        logger.info("Starting portrait generation with Google Imagen")

        ratio = self._map_aspect_ratio_to_ratio(aspect_ratio)
        logger.info(f"Using aspect ratio: {ratio}")

        try:
            # Используем Imagen через генеративный API
            model = genai.GenerativeModel(self.model_name)

            # Генерируем изображение
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.4,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )

            # Если в ответе есть изображение, извлекаем его
            if hasattr(response, 'parts') and len(response.parts) > 0:
                for part in response.parts:
                    if hasattr(part, 'inline_data'):
                        logger.info("Portrait generation completed successfully")
                        return part.inline_data.data

            # Если Imagen 3.0 не поддерживается напрямую через GenerativeModel,
            # используем альтернативный подход через REST API
            logger.warning("Direct image generation not available, using alternative method")
            return self._generate_via_rest_api(prompt, ratio)

        except Exception as e:
            logger.error(f"Error generating portrait: {e}")
            logger.info("Attempting fallback method...")
            try:
                return self._generate_via_rest_api(prompt, ratio)
            except Exception as fallback_error:
                logger.error(f"Fallback method also failed: {fallback_error}")
                raise

    def _generate_via_rest_api(self, prompt: str, aspect_ratio: str) -> bytes:
        """
        Fallback method using REST API for image generation

        Note: This is a placeholder for REST API implementation.
        In production, you would call Imagen API directly via HTTP.
        """
        import requests

        # Construct Imagen API endpoint
        # Note: Update this URL based on actual Imagen API endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateImage"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": config.GOOGLE_API_KEY
        }

        payload = {
            "prompt": prompt,
            "aspectRatio": aspect_ratio,
            "numberOfImages": 1
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()

        # Extract image data from response
        if 'images' in result and len(result['images']) > 0:
            image_data = result['images'][0]['data']
            return base64.b64decode(image_data)

        raise ValueError("No image data in API response")

    def save_image(self, image_data: bytes, save_path: str) -> bool:
        """
        Save generated image data to file

        Args:
            image_data: Image data as bytes
            save_path: Path where to save the image

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Saving image to: {save_path}")

            with open(save_path, 'wb') as f:
                f.write(image_data)

            logger.info(f"Image saved successfully")
            return True

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
