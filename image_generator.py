"""
Image Generation Module using Google Gemini
Generates AI portraits based on detailed prompts
"""
import logging
import base64
import io
import time
from typing import Optional
from PIL import Image
from google import genai
import config

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generates images using Google Gemini"""

    def __init__(self):
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self.model_name = config.GOOGLE_IMAGE_MODEL

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

    def generate_portrait(
        self,
        prompt: str,
        person_image_path: str,
        aspect_ratio: str = '1:1',
        number_of_images: int = 1
    ) -> Optional[bytes]:
        """
        Generate AI portrait using Google Gemini with person's photo

        Args:
            prompt: Detailed prompt for image generation
            person_image_path: Path to the person's photo
            aspect_ratio: Desired aspect ratio (1:1, 16:9, 9:16, etc.)
            number_of_images: Number of images to generate (default 1)

        Returns:
            Image data as bytes, or None if failed
        """
        try:
            logger.info("Starting portrait generation with Google Gemini")
            logger.info(f"Prompt length: {len(prompt)} chars")
            logger.info(f"Aspect ratio: {aspect_ratio}")

            # Load and compress the person's image for faster processing
            person_image = self.compress_image(person_image_path, max_size=1024)

            # Map aspect_ratio to pixel dimensions for explicit sizing
            aspect_map = {
                "1:1": "1024x1024",
                "16:9": "1024x576",
                "9:16": "576x1024",
                "4:3": "1024x768",
                "3:4": "768x1024",
                "3:2": "1024x683",
                "2:3": "683x1024",
            }
            pixel_size = aspect_map.get(aspect_ratio, "1024x1024")

            # Simplified generation prompt with correct priority order: STYLE → COMPOSITION → FACE
            generation_prompt = (
                f"STYLE: {prompt}\n\n"
                f"COMPOSITION: {aspect_ratio} format. "
                f"Camera angle and pose from description above, NOT from attached photo.\n\n"
                f"FACE: Use attached photo for facial identity only. "
                f"Person must be 100% recognizable."
            )

            logger.info("Calling Gemini API for image generation...")
            logger.info(f"Setting aspect ratio to: {aspect_ratio}")

            # Convert person image to base64 for new SDK
            image_b64 = self._image_to_base64(person_image)

            # Retry logic for handling temporary API overload (503, 429)
            max_retries = 3
            retry_delays = [2, 4, 8]  # Exponential backoff: 2s, 4s, 8s
            response = None

            for attempt in range(max_retries):
                try:
                    logger.info(f"API call attempt {attempt + 1}/{max_retries}")

                    # Generate content with NEW SDK including imageConfig for aspect_ratio control
                    # This is the key feature that requires the new google-genai SDK
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents={
                            'parts': [
                                {'text': generation_prompt},
                                {'inline_data': {'mime_type': 'image/jpeg', 'data': image_b64}}
                            ]
                        },
                        config={
                            'image_config': {
                                'aspect_ratio': aspect_ratio  # CRITICAL: aspect_ratio from reference style!
                            }
                        }
                    )

                    # If successful, break out of retry loop
                    logger.info("API call successful")
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
                return None

            # Extract generated image from response
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response has candidates: {hasattr(response, 'candidates')}")

            # Check if response has image data
            if not hasattr(response, 'candidates') or not response.candidates:
                logger.error("No candidates in response")
                logger.error(f"Response: {response}")
                return None

            # Get first candidate
            candidate = response.candidates[0]
            logger.info(f"Candidate type: {type(candidate)}")
            logger.info(f"Candidate has content: {hasattr(candidate, 'content')}")

            # Check for inline_data in parts (image response)
            if hasattr(candidate, 'content') and candidate.content:
                logger.info(f"Content type: {type(candidate.content)}")
                logger.info(f"Content has parts: {hasattr(candidate.content, 'parts')}")

                if hasattr(candidate.content, 'parts'):
                    logger.info(f"Number of parts: {len(candidate.content.parts)}")

                    for i, part in enumerate(candidate.content.parts):
                        logger.info(f"Part {i} type: {type(part)}")
                        logger.info(f"Part {i} has inline_data: {hasattr(part, 'inline_data')}")

                        if hasattr(part, 'inline_data') and part.inline_data:
                            logger.info(f"inline_data type: {type(part.inline_data)}")
                            logger.info(f"inline_data has data: {hasattr(part.inline_data, 'data')}")

                            # New SDK returns data - check if it's bytes or base64 string
                            image_data = part.inline_data.data
                            logger.info(f"Raw image_data type before processing: {type(image_data)}")

                            # If it's a string, decode from base64; if bytes, use directly
                            if isinstance(image_data, str):
                                logger.info("Image data is string, decoding from base64")
                                image_data = base64.b64decode(image_data)
                            else:
                                logger.info("Image data is already bytes")

                            logger.info(f"Final image data type: {type(image_data)}, length: {len(image_data)}")
                            return image_data

            logger.error("No image data found in response")
            return None

        except Exception as e:
            logger.error(f"Error generating portrait: {e}", exc_info=True)
            return None

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
