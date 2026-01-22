"""
AI Portrait Telegram Bot
Main bot logic with conversation flow
"""
import logging
import os
import tempfile
from typing import Dict
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    filters,
    ContextTypes,
)

import config
from image_analyzer import ImageAnalyzer
from image_generator import ImageGenerator

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
WAITING_REFERENCE = 0
WAITING_PERSON = 1

# User data keys
USER_REFERENCE_PATH = 'reference_path'
USER_PERSON_PATH = 'person_path'


class AIPortraitBot:
    """Main bot class"""

    def __init__(self):
        self.analyzer = ImageAnalyzer()
        self.generator = ImageGenerator()

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle /start command"""
        logger.info(f"User {update.effective_user.id} started the bot")
        await update.message.reply_text(config.MESSAGES['start'])
        return WAITING_REFERENCE

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        await update.message.reply_text(config.MESSAGES['help'])

    async def cancel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle /cancel command"""
        logger.info(f"User {update.effective_user.id} cancelled the operation")

        # Clean up temporary files
        self._cleanup_user_data(context.user_data)

        await update.message.reply_text(config.MESSAGES['cancelled'])
        return ConversationHandler.END

    async def receive_reference_image(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle reference image upload"""
        logger.info(f"User {update.effective_user.id} sent reference image")

        # Validate that it's an image
        if not update.message.photo and not update.message.document:
            await update.message.reply_text(config.MESSAGES['invalid_image'])
            return WAITING_REFERENCE

        try:
            # Download the image
            if update.message.photo:
                file = await update.message.photo[-1].get_file()
            else:
                # Check if document is an image
                if not update.message.document.mime_type in config.SUPPORTED_IMAGE_FORMATS:
                    await update.message.reply_text(config.MESSAGES['invalid_image'])
                    return WAITING_REFERENCE
                file = await update.message.document.get_file()

            # Save to temporary file
            temp_dir = tempfile.gettempdir()
            reference_path = os.path.join(
                temp_dir,
                f"reference_{update.effective_user.id}_{file.file_unique_id}.jpg"
            )
            await file.download_to_drive(reference_path)

            # Store path in user data
            context.user_data[USER_REFERENCE_PATH] = reference_path

            logger.info(f"Reference image saved: {reference_path}")

            # Send confirmation and ask for person photo
            await update.message.reply_text(config.MESSAGES['reference_received'])

            return WAITING_PERSON

        except Exception as e:
            logger.error(f"Error receiving reference image: {e}")
            await update.message.reply_text(
                config.MESSAGES['error'].format(error=str(e))
            )
            return WAITING_REFERENCE

    async def receive_person_image(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle person image upload and start processing"""
        logger.info(f"User {update.effective_user.id} sent person image")

        # Validate that it's an image
        if not update.message.photo and not update.message.document:
            await update.message.reply_text(config.MESSAGES['invalid_image'])
            return WAITING_PERSON

        try:
            # Download the image
            if update.message.photo:
                file = await update.message.photo[-1].get_file()
            else:
                # Check if document is an image
                if not update.message.document.mime_type in config.SUPPORTED_IMAGE_FORMATS:
                    await update.message.reply_text(config.MESSAGES['invalid_image'])
                    return WAITING_PERSON
                file = await update.message.document.get_file()

            # Save to temporary file
            temp_dir = tempfile.gettempdir()
            person_path = os.path.join(
                temp_dir,
                f"person_{update.effective_user.id}_{file.file_unique_id}.jpg"
            )
            await file.download_to_drive(person_path)

            # Store path in user data
            context.user_data[USER_PERSON_PATH] = person_path

            logger.info(f"Person image saved: {person_path}")

            # Send processing message
            await update.message.reply_text(config.MESSAGES['person_received'])

            # Start processing
            await self.process_and_generate(update, context)

            return WAITING_REFERENCE

        except Exception as e:
            logger.error(f"Error receiving person image: {e}")
            await update.message.reply_text(
                config.MESSAGES['error'].format(error=str(e))
            )
            return WAITING_PERSON

    async def process_and_generate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process images and generate AI portrait"""
        try:
            reference_path = context.user_data.get(USER_REFERENCE_PATH)
            person_path = context.user_data.get(USER_PERSON_PATH)

            if not reference_path or not person_path:
                raise ValueError("Missing image paths")

            logger.info("Starting image analysis...")

            # Step 1: Analyze reference image (only for aspect_ratio)
            logger.info("Analyzing reference image for aspect ratio...")
            reference_data = self.analyzer.analyze_reference_image(reference_path)
            aspect_ratio = reference_data.get('aspect_ratio', '1:1')

            logger.info(f"Aspect ratio: {aspect_ratio}")

            # Send progress update
            await update.message.reply_text(config.MESSAGES['processing'])

            # Step 2: Generate portrait with 2 images
            logger.info("Generating portrait with Google Gemini...")
            logger.info("Passing reference image for style and person image for identity")
            image_data = self.generator.generate_portrait(
                reference_image_path=reference_path,
                person_image_path=person_path,
                aspect_ratio=aspect_ratio
            )

            if not image_data:
                raise ValueError("Failed to generate image")

            # Save generated image
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(
                temp_dir,
                f"portrait_{update.effective_user.id}_{os.urandom(4).hex()}.png"
            )
            self.generator.save_image(image_data, output_path)

            logger.info(f"Portrait generated and saved: {output_path}")

            # Send the generated portrait
            with open(output_path, 'rb') as photo:
                await update.message.reply_photo(
                    photo=photo,
                    caption=config.MESSAGES['success']
                )

            # Clean up all temporary files
            self._cleanup_user_data(context.user_data)
            try:
                os.remove(output_path)
            except Exception as e:
                logger.warning(f"Failed to remove output file: {e}")

            logger.info("Portrait generation completed successfully")

        except Exception as e:
            logger.error(f"Error processing and generating portrait: {e}", exc_info=True)
            await update.message.reply_text(
                config.MESSAGES['error'].format(error=str(e))
            )
            # Clean up on error
            self._cleanup_user_data(context.user_data)

    def _cleanup_user_data(self, user_data: Dict) -> None:
        """Clean up temporary files"""
        for key in [USER_REFERENCE_PATH, USER_PERSON_PATH]:
            if key in user_data:
                try:
                    path = user_data[key]
                    if os.path.exists(path):
                        os.remove(path)
                        logger.info(f"Removed temporary file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")
                finally:
                    del user_data[key]


def main():
    """Start the bot"""
    logger.info("Starting AI Portrait Telegram Bot...")

    # Initialize bot
    bot = AIPortraitBot()

    # Create application
    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Create conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', bot.start_command)],
        states={
            WAITING_REFERENCE: [
                MessageHandler(
                    filters.PHOTO | filters.Document.IMAGE,
                    bot.receive_reference_image
                )
            ],
            WAITING_PERSON: [
                MessageHandler(
                    filters.PHOTO | filters.Document.IMAGE,
                    bot.receive_person_image
                )
            ],
        },
        fallbacks=[
            CommandHandler('cancel', bot.cancel_command),
            CommandHandler('start', bot.start_command),
        ],
    )

    # Register handlers
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('help', bot.help_command))

    # Start bot with webhook (for production deployment on Render)
    logger.info(f"Starting webhook on {config.WEBHOOK_URL}")
    logger.info(f"Listening on port {config.PORT}")

    application.run_webhook(
        listen="0.0.0.0",
        port=config.PORT,
        url_path="webhook",
        webhook_url=f"{config.WEBHOOK_URL}/webhook"
    )


if __name__ == '__main__':
    main()
