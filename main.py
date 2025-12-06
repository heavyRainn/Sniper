# main.py
from bot import TrendDivBot
from config import BYBIT_KEY, BYBIT_SECRET
from logger_setup import setup_logger
from utils import mask_secret


def main():
    logger = setup_logger("trend_div_bot")

    logger.info(
        "Starting bot with KEY=%s SECRET=%s",
        mask_secret(BYBIT_KEY),
        mask_secret(BYBIT_SECRET),
    )

    bot = TrendDivBot()
    bot.run()


if __name__ == "__main__":
    main()
