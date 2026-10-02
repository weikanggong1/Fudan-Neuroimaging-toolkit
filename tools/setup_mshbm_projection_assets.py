"""从 Git checkout 部署 MS-HBM 的固定 MNI 体积投影资源。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fnit.mshbm.assets_setup import main

if __name__ == "__main__":
    main()
