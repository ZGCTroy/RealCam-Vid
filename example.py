from pathlib import Path
from pprint import pprint

from torch import Tensor

from scripts.i2v_camera_dataset import I2VDatasetWithResize

data_root = "/mnt/nfs/data/datasets/RealCam-Vid"
dataset = I2VDatasetWithResize(
    data_root=Path(data_root),
    metadata_path=Path(f"{data_root}/RealCam-Vid_test.npz"),
    max_num_frames=81,
    width=1360,
    height=768,
    keep_aspect_ratio=True,
)

for data in dataset:
    pprint({k: v.shape if isinstance(v, Tensor) else v for k, v in data.items()}, sort_dicts=False)
    break
