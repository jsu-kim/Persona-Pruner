import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_pruner.data_pipeline.rewrite import build_rewrite_prompts_main


if __name__ == "__main__":
    build_rewrite_prompts_main()
