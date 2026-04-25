import sys
import os

# Ensure go_engine/ is on the path so `from engine.xxx import ...` works
# whether pytest is invoked from go_engine/ or from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
