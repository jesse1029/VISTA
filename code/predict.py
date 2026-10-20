"""Portable frozen-model CLI. No DESIS or crop-label argument exists."""
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vista import FrozenVISTA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="N-by-59 PRISMA .npy file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = FrozenVISTA.load(args.model).predict(np.load(args.input, allow_pickle=False))
    np.savez_compressed(args.output, spectra=result.spectra, fallback=result.fallback,
                        committed=result.committed, annotation_index=result.annotation_index)

if __name__ == "__main__":
    main()
