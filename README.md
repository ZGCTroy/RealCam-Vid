# RealCam-Vid Dataset

<div align="center">
    <!-- <a href="https://arxiv.org/abs/2410.15957"><img src="https://img.shields.io/static/v1?label=arXiv&message=2410.15957&color=b21d1a"></a> -->
    <a href="https://github.com/ZGCTroy/RealCam-Vid"><img src="https://img.shields.io/static/v1?label=GitHub&message=Repo&color=green"></a>
    <a href="https://huggingface.co/datasets/MuteApo/RealCam-Vid"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=Dataset&color=blue"></a>
</div>

## News

- 25/04/08: We provide torch dataset [demo code](example.py) for example usage of our RealCam-Vid.
- 25/03/26: Release our dataset [RealCam-Vid](https://huggingface.co/datasets/MuteApo/RealCam-Vid) v1 for metric-scale camera-controlled video generation, containing ~100K video clips with dedicated short/long captions and metric-scale camera annotations.
- 25/02/18: Initial commit of the project, we plan to release the full dataset and data processing code in several weeks. DiT-based models (e.g., CogVideoX) trained on our dataset will be available at [RealCam-I2V](https://github.com/ZGCTroy/RealCam-I2V).

## Motivation

Current datasets for camera-controllable video generation face critical limitations that hinder the development of robust and versatile models. 
Our curated dataset and data-processing pipeline uniquely combines **diverse scene dynamics** with **metric-scale camera trajectories**, enabling generative models to learn both scene dynamics and camera motion in a unified framework.

### Training Data Variation

<table>
    <tr>
        <td align="center"><img src="https://github.com/user-attachments/assets/7d8ff359-8e31-4db4-838e-79061cffd651"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/9c65a8c0-77e8-4fbe-903b-3b6ab7492983"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/21e5cb60-b639-41ce-9a9e-477dd02500b1"></td>
    </tr>
    <tr>
        <td align="center">Static Scene<br>&<br>Dynamic Camera</td>
        <td align="center">Dynamic Scene<br>&<br>Static Camera</td>
        <td align="center">Dynamic Scene<br>&<br>Dynamic Camera</td>
    </tr>
</table>

Existing datasets for camera motions and scene dynamics suffer from **domain-specific biases** and **incomplete annotations**, limiting their utility for training robust real-world models.
- **Static Scene & Dynamic Camera datasets** (e.g., RealEstate10K, DL3DV) 
    - **Strengths**: High aesthetic quality, dense relative-scale camera trajectory annotations.
    - **Weaknesses**: Static scenes lack object dynamics, leading to models that fail to generalize to real-world dynamic environments due to overfitting to rigid structures.
- **Dynamic Scene & Static Camera datasets** (e.g., 360-Motion)
    - **Strengths**: Capture dynamic objects.
    - **Weaknesses**: Omit camera motion, limiting their utility for trajectory-based video generation.
- **Dynamic Scene & Dynamic Camera datasets** (e.g., MiraData) 
    - **Strengths**: Exhibit rich real-world dynamics (moving objects + camera motion).
    - **Weaknesses**: No metric-scale camera annotations, making them unsuitable for metric-scale training.


### Camera Pose Annotation

<table>
    <tr>
        <td align="center"><img src="https://github.com/user-attachments/assets/6145e3b1-00ff-4701-95c1-e74b98fb8ad2"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/7f864853-5587-4f7b-b43f-e1cd93830bee"></td>
    </tr>
    <tr>
        <td align="center">Real-world Video</td>
        <td align="center">4D Recontruction</td>
    </tr>
</table>

Our pipeline leverages [**MonST3R**](https://github.com/Junyi42/monst3r) to provide **high-quality camera trajectory annotations for videos with dynamic scenes**. Unlike SLAM/COLMAP, which rely on keypoint matches vulnerable to dynamic outliers, this state-of-the-art method explicitly models per-frame geometry while distinguishing moving objects from static scenes.
- Current camera annotation methods, such as those used in RealEstate10K and DL3DV, rely heavily on SLAM (Simultaneous Localization and Mapping) and COLMAP (Structure-from-Motion). These methods are primarily designed for **static scenes**, where the environment remains unchanged during the capture process.
- In real-world videos, **dynamic foreground objects** (e.g., moving people, vehicles) introduce noise into the feature matching process. These objects create inconsistent feature tracks, leading to errors in camera pose estimation and 3D reconstruction.


### Metric Scale Scene Alignment

<table>
    <tr>
        <td align="center"><img src="https://github.com/user-attachments/assets/1d6e3b51-2d1f-4c61-b64a-303a0a0937a6"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/8f9a373e-9b9d-4be1-9066-71bd5e383de6"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/325a9acc-33f8-4fae-8366-aa465fd1edba"></td>
    </tr>
    <tr>
        <td align="center">GT Video</td>
        <td align="center">Unaligned Preview<br>(Relative Scale)</td>
        <td align="center">Aligned Preview<br>(Metric Scale)</td>
    </tr>
</table>

Aligning camera trajectories to a metric scale is critical when constructing datasets from heterogeneous sources (e.g., RealEstate10K, DL3DV, MiraData).
- **Cross-Dataset Compatibility**: Relative scales differ across datasets (e.g., "1 unit" in RealEstate10K ≠ "1 unit" in MiraData), causing misalignment and **scale ambiguity** in 3D reconstructions or motion priors.
- **Real-World Applicability**: Metric-scale alignment (e.g., meters) ensures consistency for training and evaluation, enabling models to learn **physically meaningful motion patterns** (e.g., velocity in m/s).
- **Enhanced Physical Consistency**: Scene dimensions (e.g., room sizes, object heights) match real-world proportions, critical for tasks like 3D reconstruction or object interaction modeling as **geometric correctness**.

### Data Source

#### DL3DV-10K

<table>
    <tr>
        <td align="center"><img src="https://github.com/user-attachments/assets/e5309aac-285d-4ee4-803f-d1103379522e"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/c3e6b49d-fbfd-45d6-95e4-3657c81beb52"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/b9dab160-3a31-45f6-8643-72faa9b852a0"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/d73f9a02-6608-4e1e-8a58-0346da4205c3"></td>
    </tr>
    <tr>
        <td align="center">Hotels & Accommodations</td>
        <td align="center">Education Institutions</td>
        <td align="center">Restaurants & Cafes</td>
        <td align="center">Sports & Fitness</td>
    </tr>
    <tr>
        <td align="center"><img src="https://github.com/user-attachments/assets/d3c3f739-e128-48ee-b12a-5951d1a4ac14"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/464783aa-2bd8-45aa-9f93-56e23e374cd7"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/ebb0b459-cd1f-4c4d-94bd-8cf7d9985803"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/5abc2523-69cd-486b-9717-caba6ee91cab"></td>
    </tr>
    <tr>
        <td align="center">Shopping Centers</td>
        <td align="center">Cultural Centers</td>
        <td align="center">Art Galleries</td>
        <td align="center">Lakes</td>
    </tr>
</table>

#### MiraData

<table>
    <tr>
        <td align="center"><img src="https://github.com/user-attachments/assets/d82a2ca4-3ea8-42b8-91d6-723c50959f43"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/1c8f5b7c-b09b-45c5-a6ff-c4fd3bb8b950"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/fbf5c100-4be2-4c20-9ab5-09b6d15f28b2"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/0497545e-5965-4e79-98a4-4b770fca171a"></td>
    </tr>
    <tr>
        <td align="center">Game Renderings</td>
        <td align="center">Sports</td>
        <td align="center">FPV Drones</td>
        <td align="center">City Explorations</td>
    </tr>
</table>

#### RealEstate10K

## Guideline

### Download

We recommend [hfd.sh](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f) for downloading.

```shell
wget https://gist.githubusercontent.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f/raw/6891c4b02f5cf3d014c7b1523556e15d9a3dd00f/hfd.sh
chmod u+x hfd.sh
hfd.sh MuteApo/RealCam-Vid --dataset \
    --hf_username <YOUR_HF_USER_NAME> --hf_token <YOUR_HF_ACCESS_TOKEN>
```

Unzip downloaded zip files.

```shell
cd RealCam-Vid
find "zip" -type f -name "*.zip" -exec sh -c '
    for zip_file do
        unzip $zip_file
    done
' sh {} +
```

### Metadata Format

We split our dataset into 2 split, namely train set (~100K) and test set (5K).
The train/test metadata `npz` file contains a list of dicts with following key fields for each video clip:

- `video_path`: relative path of the video clip with respect to the data root folder.
- `short_caption`: short caption (<=77 tokens) provided by [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), which is suitable for CLIP series as text encoders (U-Net based models).
- `long_caption`: long caption (<=226 tokens) provided by [CogVLM2-Caption](https://github.com/THUDM/CogVideo/tree/main/tools/caption), which is suitable for T5 series as text encoders (DiT based models).
- `camera_intrinsics`: quadruple of camera intrinsics in order of `fx`, `fy`, `cx`, `cy`, and their values are normalized by the corresponding `w`/`h` to accommodate varying video resolutions.
- `camera_extrinsics`: 4x4 relative-scale world-to-camera (w2c) matrices under OpenCV/COLMAP camera convention.
- `align_factor`: scale factor for w2c alignment from relative-scale to metric-scale.
- `camera_scale`: maximum L2 distance across all camera positions of the video clip from the first frame.
- `vtss_score`: Video Training Suitability Score measured by [Koala-36M](https://github.com/KwaiVGI/Koala-36M).

### Example Usage

We provide a basic torch dataset in [i2v_camera_dataset.py](scripts/i2v_camera_dataset.py) to load RealCam-Vid metadata `npz`, including videos, captions and camera annoations. Please refer to the code for more details.

### Ethics Concerns

All videos of RealCam-Vid dataset are sourced from public domains, and are intended solely for informational purposes only.
The copyright remains with the original owners of the video.
Our institution are not responsible for the content nor the meaning of these videos.
If you have any concerns, please contact [us](mailto:guangcongzheng\@zju.edu.cn) and we will promptly remove them.


### Related Projects

- [RealEstate10K](https://google.github.io/realestate10k)
- [DL3DV](https://dl3dv-10k.github.io/DL3DV-10K)
- [MiraData](https://mira-space.github.io)
- [MonST3R](https://monst3r-project.github.io)
- [RealCam-I2V](https://zgctroy.github.io/RealCam-I2V)
- [CamI2V](https://zgctroy.github.io/CamI2V)

### Citations

```
@article{li2025realcam,
    title={RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control}, 
    author={Li, Teng and Zheng, Guangcong and Jiang, Rui and Zhan, Shuigen and Wu, Tao and Lu, Yehao and Lin, Yining and Li, Xi},
    journal={arXiv preprint arXiv:2502.10059},
    year={2025},
}

@article{zheng2024cami2v,
    title={CamI2V: Camera-Controlled Image-to-Video Diffusion Model},
    author={Zheng, Guangcong and Li, Teng and Jiang, Rui and Lu, Yehao and Wu, Tao and Li, Xi},
    journal={arXiv preprint arXiv:2410.15957},
    year={2024}
}
```
