import sys
from pathlib import Path

# scripts/ is not a package; make its modules importable the same way the
# EDA percent-script does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
