"""
Image Generation Module using Google Gemini
Generates AI portraits based on detailed prompts
"""
import logging
import base64
from typing import Optional
from PIL import Image
import google.generativeai as genai
import config

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generates images using Google Gemini"""

    def __init__(self):
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(config.GOOGLE_IMAGE_MODEL)

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

            # Load the person's image
            person_image = Image.open(person_image_path)

            # Create the generation prompt with critical instructions
            generation_prompt = (
                f"MASTERPIECE RENDERING. CRITICAL: Respect the MEDIUM identified in the prompt. "
                f"If it's a photo, make it look like a physical print. "
                f"If it's art, show the physical texture of paper/canvas. "
                f"LIKENESS IS MANDATORY. EXECUTE PROMPT: {prompt}"
            )

            logger.info("Calling Gemini API for image generation...")

            # Generate content with image and prompt
            # Note: aspect_ratio is handled in the prompt, Gemini will return image in response
            response = self.model.generate_content(
                [person_image, generation_prompt]
            )

            # Extract generated image from response
            if not response.candidates or len(response.candidates) == 0:
                logger.error("No candidates in response")
                logger.error(f"Response: {response}")
                return None

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                logger.error("No content parts in response")
                logger.error(f"Candidate: {candidate}")
                return None

            logger.info(f"Response has {len(candidate.content.parts)} parts")

            # Find the image in response parts
            for i, part in enumerate(candidate.content.parts):
                logger.info(f"Part {i}: has inline_data={hasattr(part, 'inline_data')}")
                if hasattr(part, 'inline_data') and part.inline_data:
                    logger.info("Successfully extracted generated image")
                    image_data = part.inline_data.data
                    logger.info(f"Image data type: {type(image_data)}, length: {len(image_data) if image_data else 0}")
                    return image_data

            logger.error("No image data found in response parts")
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
