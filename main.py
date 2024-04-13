import argparse
from training.miaembtrainer import MiaEmbeddingsTrainer
from utils import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath', type=str, required=True, help='Training configuration file path (path + filename)')
    args = parser.parse_args()

    trainer = MiaEmbeddingsTrainer(args.config_filepath)
    trainer.train()
