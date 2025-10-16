import os
from argparse import ArgumentParser

scenes = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]
exp_name = "nerfsyn"
output_dir = f"./output/{exp_name}"
dataset_dir = "/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic"

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--yaml", default="./configs/xxx.yaml")
args, _ = parser.parse_known_args()

device = 0

yaml_file = args.yaml
print(f"read yaml file from : ", yaml_file)

if not args.skip_training:
    for scene in scenes:
        source = f'{dataset_dir}/{scene}'
        save_dir = f'{output_dir}/{scene}'
        ip = f"127.0.0.{device}"
        ope = f'CUDA_VISIBLE_DEVICES={device} python train.py -s {source} --eval --scene_name {scene} --yaml {yaml_file} -m {save_dir} --ip {ip}'

        print("running: \n", ope)
        os.system(ope)

if not args.skip_rendering:
    common_args = " --skip_train --skip_mesh "
    for scene in scenes:
        source = f'{dataset_dir}/{scene}'
        save_dir = f'{output_dir}/{scene}'
        ope = f"python eval_render.py --iteration 30000 -s {source} -m {save_dir} --yaml {yaml_file} " + common_args

        os.system(ope)

if not args.skip_metrics:
    ope = f"python scripts/metric_eval.py {output_dir} {30000}"
    os.system(ope)