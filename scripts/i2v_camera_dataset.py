from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import (
    load_images_from_videos,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")


class I2VDatasetWithResize(Dataset):
    def __init__(
        self,
        data_root: Path,
        metadata_path: Path,
        max_num_frames: int,
        height: int,
        width: int,
        keep_aspect_ratio: bool,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.all_metadata = np.load(metadata_path, allow_pickle=True)["arr_0"]

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        self.keep_aspect_ratio = keep_aspect_ratio

        self.__transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])

    def __len__(self) -> int:
        return len(self.all_metadata)

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        metadata = self.all_metadata[index]
        video_path = self.data_root.joinpath(metadata["video_path"])
        image_path = load_images_from_videos([video_path])[0]
        prompt = metadata["long_caption"]
        align_factor = metadata["align_factor"]
        fx, fy, cx, cy = metadata["camera_intrinsics"]
        w2cs = torch.from_numpy(metadata["camera_extrinsics"])  # [F, 4, 4]

        video, indices = preprocess_video_with_resize(
            video_path, self.max_num_frames, self.height, self.width, keep_aspect_ratio=self.keep_aspect_ratio
        )
        image = preprocess_image_with_resize(
            image_path, self.height, self.width, keep_aspect_ratio=self.keep_aspect_ratio
        )
        if self.keep_aspect_ratio:
            video, resized_H, resized_W = self.resize_for_rectangle_crop(video, self.height, self.width)
            image, _, _ = self.resize_for_rectangle_crop(image.unsqueeze(0), self.height, self.width)
        else:
            resized_H, resized_W = video.shape[-2:]

        cur_H, cur_W = video.shape[-2:]
        video = self.video_transform(video)
        image = self.image_transform(image)

        camera_intrinsics = torch.tensor(
            [
                [
                    [fx * resized_W, 0, cx * cur_W],
                    [0, fy * resized_H, cy * cur_H],
                    [0, 0, 1],
                ]
            ]
            * video.shape[0]
        )

        w2cs_sampled = w2cs[indices]
        c2ws_sampled = w2cs_sampled.inverse()
        c2ws_sampled[:, :3, 3] *= align_factor
        camera_extrinsics = c2ws_sampled.inverse()

        return {
            "prompt": prompt,
            "image": image,  # C, H, W
            "video": video,  # F, C, H, W
            "camera_intrinsics": camera_intrinsics,  # F, 3, 3
            "camera_extrinsics": camera_extrinsics,  # F, 4, 4
        }

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__transforms(f) for f in frames], dim=0)

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__transforms(image)

    def resize_for_rectangle_crop(self, frames, H, W):
        ori_H, ori_W = frames.shape[-2:]

        if ori_W / ori_H > W / H:
            frames = transforms.functional.resize(frames, size=[H, int(ori_W * H / ori_H)])
        else:
            frames = transforms.functional.resize(frames, size=[int(ori_H * W / ori_W), W])

        resized_H, resized_W = frames.shape[2], frames.shape[3]
        frames = frames.squeeze(0)

        delta_H = resized_H - H
        delta_W = resized_W - W

        top, left = delta_H // 2, delta_W // 2
        frames = transforms.functional.crop(frames, top=top, left=left, height=H, width=W)

        return frames, resized_H, resized_W
