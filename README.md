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
