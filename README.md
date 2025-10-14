<h1 align="center"> Neural Shell Texture Splatting</h1>
<p align="center"><b>ICCV 2025</b></p>
<p align="center"><a href="https://zhangxin-cg.github.io/nest-splatting/" target="_blank">Project Page</a> | <a href="https://arxiv.org/pdf/2507.20200" target="_blank">Paper</a></p>

<p align="center">
<a href="https://zhangxin-cg.github.io/" target="_blank">Xin Zhang<sup>1</sup></a>, 
<a href="https://apchenstu.github.io/" target="_blank">Anpei Chen<sup>2</sup></a>, 
<a href="https://venite-xjc.github.io/" target="_blank">Jincheng Xiong<sup>1</sup></a>, 
<a href="https://turandai.github.io/" target="_blank">Pinxuan Dai<sup>1</sup></a>, 
<a href="https://shenyujun.github.io/" target="_blank">Yujun Shen<sup>3</sup></a>, 
<a href="http://www.cad.zju.edu.cn/home/weiweixu/index.htm" target="_blank">Weiwei Xu<sup>1†</sup></a>
</p>

<p align="center"><sup>1</sup> Zhejiang University, <sup>2</sup> Westlake University, <sup>3</sup> Ant Group</p>

![teaser](/assets/teaser.png)

## Notes

This is the official code for our paper *Neural Shell Texture Splatting: More Details and Fewer Primitives*. Our work primarily focuses on decoupling the inherent geometry and texture representation in Gaussian splatting. Specifically, we use **2D Gaussian splats** as the geometric representation and adopt a **multi-level instant hash table** to encode the texture field. This enables richer texture rendering with fewer Gaussian points.

Our code framework is primarily based on **[2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)** and **[instant-ngp](https://github.com/NVlabs/instant-ngp)**.

Our method's core implementation includes:
1. Computing the **ray-splat intersection** during Gaussian rasterization and querying the texture features of the intersection point from the hash table.
2. Deferred rendering generates the feature map, which is then decoded into an RGB image by incorporating the ray direction.


## Setup
The code was tested on:
- **OS**: Ubuntu 22.04.5 LTS
- **GPU**: NVIDIA RTX 4090
- **CUDA Version**: 11.8
- **Python Version**: 3.10
- **Torch Version**: 2.1.0

To create a conda environment with the same dependencies as the codebase, please run:
```bash
# download
git clone https://github.com/zhangxin-cg/nest-splatting.git --recursive
cd nest-splatting

# conda env
conda env create -f environment.yml
conda activate nest_splatting

# install tcnn extension
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Quick Examples

Assume you have downloaded [NeRF synthetic](https://drive.google.com/file/d/1OsiBs2udl32-1CqTXCitmov4NQCYdA9g/view?usp=share_link), simply run
```bash
python scripts/nerfsyn_eval.py --yaml ./configs/nerfsyn.yaml
```
With the [MipNeRF360 dataset](https://jonbarron.info/mipnerf360/), you can run
```bash
python scripts/360_eval.py
```
With the [DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36), you can run
```bash
python scripts/dtu_eval.py --yaml ./configs/dtu.yaml 
```
Remember to set the `dataset_dir` (and `dtu_data` for geometric comparison) in the scripts to the actual dataset location before running the commands.

**Custom Dataset:** We use the same COLMAP loader as 3DGS, you can prepare your data following [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes).
Create a YAML config for your scene and check the bounding-box size(`range` in .yaml) as well as background contract flag (`contract` in .yaml) for outdoor data.

## License

This codebase is Apache 2.0 licensed. Please refer to the [LICENSE](LICENSE) file for more details.

## Citation
If this project supports your research, please consider citing:

```bibtex
@inproceedings{zhang2025nest,
  title = {Neural Shell Texture Splatting: More Details and Fewer Primitives},
  author = {Zhang, Xin and Chen, Anpei and Xiong, Jincheng and Dai, Pinxuan and Shen, Yujun and Xu, Weiwei},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2025}
}

```