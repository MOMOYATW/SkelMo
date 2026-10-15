# SkelMo: Universal Skeletal Motion Generation for 3D Rigged Shapes

[Project page](https://research.davytao.me/skelmo/) · [Paper](https://arxiv.org/abs/2606.01518)

Official implementation of **SkelMo: Universal Skeletal Motion Generation for 3D Rigged Shapes** (ECCV 2026) by Ye Tao, Yuxin Yao, Kendong Liu, Dapeng Wu, and Junhui Hou.

![SkelMo teaser](assets/teaser-eccv.png)

## Install

```bash
conda env create -f environment.yaml
conda activate skelmo
```

The first inference run downloads DINOv2 ViT-B/14 weights through `timm`.

## Checkpoint inference

Download the [checkpoint bundle](https://drive.google.com/file/d/1ZJ4BKAkn4ms21tmoiNAxJtbXDT3GmKtZ/view?usp=sharing) and follow the [inference guide](docs/CHECKPOINT_INFERENCE.md). The included renderer prepares uncropped frames, calibrated camera, and first-frame skeleton from an animated rigged GLB. An arbitrary driving video still needs its camera and first-frame 3D pose supplied separately.

<details>
<summary>Optional rest-pose appearance checkpoint</summary>

Download the [rest-pose bundle](https://drive.google.com/file/d/1yqu-rJMjgmCVTmmTlLLYAPU6B5feWBnR/view?usp=sharing) (`skelmo_rest_merged_checkpoint_bundle.tar.gz`; SHA-256 `fd089b83959af87a7c2a07847c83b283c31c771f94da09a04254b9f4c83dbcbd`). It changes only the joint appearance descriptors; the aligned first-frame skeleton and video are still required.

Prepare the same input as in the [inference guide](docs/CHECKPOINT_INFERENCE.md), adding `rest_pose [J,3]` and `rest_appearance [J,768]` in joint order. Generate the 14-view DINO descriptors with the [rest-feature pipeline](preprocess/joint_features/README.md).

```bash
mkdir -p checkpoints
tar -xzf skelmo_rest_merged_checkpoint_bundle.tar.gz -C checkpoints
CUDA_VISIBLE_DEVICES=0 python -m sample.generate_checkpoint \
  --checkpoint checkpoints/skelmo_rest_merged/model000099999.pt \
  --stats checkpoints/skelmo_rest_merged/training_stats.json \
  --pca checkpoints/skelmo_rest_merged/pca_128.npz \
  --input /path/to/rest_prepared.npz \
  --frames-dir /path/to/uncropped_frames \
  --guidance-scale 1 --output outputs/rest_example
```

</details>

The processed training dataset is not included. The checkpoint's `args.json` records its configuration.

## Citation

```bibtex
@inproceedings{tao2026skelmo,
  title     = {SkelMo: Universal Skeletal Motion Generation for 3D Rigged Shapes},
  author    = {Tao, Ye and Yao, Yuxin and Liu, Kendong and Wu, Dapeng and Hou, Junhui},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License and acknowledgements

See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). SkelMo builds on [AnyTop](https://anytop2025.github.io/Anytop-page/) and code derived from Motion Diffusion Model, OpenAI guided diffusion, GRPE, and Audiocraft; visual features use [DINOv2](https://github.com/facebookresearch/dinov2). Follow the respective dataset and component licenses.
