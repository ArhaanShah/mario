"""Write a synthetic MARIO-shaped dataset. Usage: python scripts/make_synthetic_data.py --out smoke_out/data"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mario_dino.synthetic import main  # noqa: E402

if __name__ == "__main__":
    main()
