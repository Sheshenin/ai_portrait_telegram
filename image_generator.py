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

            # Create the generation prompt with clear separation: face from photo, pose from description
            generation_prompt = (
                f"CRITICAL INSTRUCTIONS:\n\n"
                f"OUTPUT FORMAT (MANDATORY):\n"
                f"   - Generate image with EXACT aspect ratio: {aspect_ratio}\n"
                f"   - Output dimensions MUST be: {pixel_size} pixels\n"
                f"   - This aspect ratio comes from reference style image, NOT from person's photo\n"
                f"   - DO NOT use aspect ratio or dimensions from the attached photograph\n"
                f"   - The attached photo is {person_image.width}x{person_image.height} - IGNORE these dimensions\n\n"
                f"1. IDENTITY REFERENCE (Use attached photograph):\n"
                f"   - The attached photograph is ONLY an identity reference for facial likeness\n"
                f"   - USE ONLY THE FACE: facial features, proportions, unique characteristics\n"
                f"   - PRESERVE 100% facial recognition - the person MUST be recognizable\n"
                f"   - DO NOT use the pose, angle, or framing from the photograph\n"
                f"   - Think of it as: 'this person's face' transplanted into a different scene\n\n"
                f"2. COMPOSITION & POSE (Use prompt description):\n"
                f"   - IGNORE the frontal pose from the photograph\n"
                f"   - Follow EXACTLY the composition, pose, angle, and framing from the prompt below\n"
                f"   - If prompt says 'sitting on knees', do NOT use frontal standing from photo\n"
                f"   - If prompt says 'shot from above', do NOT use straight angle from photo\n"
                f"   - Camera angle, subject placement, and dynamics come from PROMPT, not photo\n\n"
                f"3. ARTISTIC STYLE:\n"
                f"   - Apply the medium and style as described in the prompt\n"
                f"   - If it's a photo, show physical print texture\n"
                f"   - If it's art, show canvas/paper texture and brushwork\n\n"
                f"EXECUTE PROMPT: {prompt}"
            )

            logger.info("Calling Gemini API for image generation...")
            logger.info(f"Setting aspect ratio to: {aspect_ratio}")

            # Convert aspect_ratio string (e.g., "9:16") to config format
            # Try multiple approaches as google-generativeai SDK documentation is unclear

            # Approach 1: Try passing aspect_ratio in generation_config
            try:
                generation_config = {
                    "temperature": 0.4,
                    "image_config": {
                        "aspect_ratio": aspect_ratio
                    }
                }
            except:
                # Fallback if image_config not supported
                generation_config = genai.types.GenerationConfig(
                    temperature=0.4,
                )

            # Generate content with image and prompt
            # aspect_ratio from reference image determines output format
            response = self.model.generate_content(
                [person_image, generation_prompt],
                generation_config=generation_config,
                request_options={"timeout": 180}
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
