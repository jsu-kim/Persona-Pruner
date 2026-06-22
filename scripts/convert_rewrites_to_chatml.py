import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_pruner.data_pipeline.rewrite import convert_rewrites_main


if __name__ == "__main__":
    convert_rewrites_main()
