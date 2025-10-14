import os
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]

scenes = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
exp_name = "mip360"
output_dir = f"./output/{exp_name}"
dataset_dir = "/workspace/DATA/360_v2"

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
# parser.add_argument("--yaml", default="./configs/2dgs.yaml")
args, _ = parser.parse_known_args()

device = 0

indoor_yaml = './configs/360_indoor.yaml'
outdoor_yaml = './configs/360_outdoor.yaml'

if not args.skip_training:
    for scene in scenes:
        common_args = ""
        if scene in mipnerf360_indoor_scenes:
            common_args += f" -i images_2 --yaml {indoor_yaml}"
        elif scene in mipnerf360_outdoor_scenes:
            common_args += f" -i images_4 --yaml {outdoor_yaml}"
        else:
            raise RuntimeError(f"Unknown scene {scene}.")

        source = f'{dataset_dir}/{scene}'
        save_dir = f'{output_dir}/{scene}'
        ip = f"127.0.0.{device}"
        ope = f'CUDA_VISIBLE_DEVICES={device} python train.py -s {source} --eval --scene_name {scene} -m {save_dir} --ip {ip}' + common_args

        print("running: \n", ope)
        os.system(ope)

if not args.skip_rendering:
    common_args = " --skip_train --skip_mesh "
    for scene in scenes:
        if scene in mipnerf360_indoor_scenes:
            common_args += f" --yaml {indoor_yaml}"
        elif scene in mipnerf360_outdoor_scenes:
            common_args += f" --yaml {outdoor_yaml}"
        else:
            raise RuntimeError(f"Unknown scene {scene}.")
        
        source = f'{dataset_dir}/{scene}'
        save_dir = f'{output_dir}/{scene}'
        ope = f"python eval_render.py --iteration {35000} -s {source} -m {save_dir}" + common_args

        os.system(ope)


if not args.skip_metrics:
    ope = f"python scripts/metric_eval.py {output_dir} {30000}"
    os.system(ope)