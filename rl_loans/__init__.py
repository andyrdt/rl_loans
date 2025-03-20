import os
import sys
from pathlib import Path

# Setup path for safety-tooling
project_root = Path(__file__).parent.parent
safety_tooling_path = str(project_root / "safety-tooling")
if safety_tooling_path not in sys.path:
    sys.path.append(safety_tooling_path)