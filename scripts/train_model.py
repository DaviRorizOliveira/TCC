import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_models.train_model import train_model

if __name__ == "__main__":
    train_model()