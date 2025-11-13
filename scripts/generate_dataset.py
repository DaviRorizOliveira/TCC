import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.dataset_generator import generate_dataset

if __name__ == "__main__":
    generate_dataset( num_samples_per_type = 5, n_range = (500, 1000), max_workers = 8, output_file = 'data/processed/train/dataset3.csv')