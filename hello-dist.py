import logging
import os
import sys

logger = logging.getLogger(__name__)

def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    logger.warning(
        f"Process rank: {env_local_rank}"
    )


if __name__ == "__main__":
    main()