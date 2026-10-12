from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'grid'))
from grid_identity import OMIT, compare, field_bits, fields
