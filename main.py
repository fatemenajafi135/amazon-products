import logging
from core.price_predictor import PricePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    categories = ['Badminton', 'Jeans', 'Televisions']
    for category in categories:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing category: {category}")
        predictor = PricePredictor(category)
        predictor.execute()
        