# Package initialization for vdu_scanner
import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))
