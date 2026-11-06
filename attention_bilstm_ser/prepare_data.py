import argparse
from src.config import DATA_DIR, FEATURE_CACHE
from src.features import build_cache

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract RAVDESS speech features")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output", default=str(FEATURE_CACHE))
    args = parser.parse_args()
    build_cache(args.data_dir, args.output)
