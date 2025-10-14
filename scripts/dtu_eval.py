import os
from argparse import ArgumentParser

dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']
exp_name = "dtu"
output_dir = f"./output/{exp_name}"
dataset_dir = "/workspace/DATA/DTU"
dtu_data = "xxx"

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--yaml", default="./configs/2dgs.yaml")
args, _ = parser.parse_known_args()

device = 0
iteration = 30000

yaml_file = args.yaml
print(f"read yaml file from : ", yaml_file)

if not args.skip_training:
    common_args = " -r 2 --eval "
    for scene in dtu_scenes:
        source = f'{dataset_dir}/{scene}'
        save_dir = f'{output_dir}/{scene}'
        ip = f"127.0.0.{device}"
        ope = f'CUDA_VISIBLE_DEVICES={device} python train.py -s {source} --scene_name {scene} -m {save_dir} --yaml {yaml_file} --ip {ip}' + common_args
        print(ope)
        os.system(ope)

if not args.skip_rendering:
    all_sources = []
    common_args = " --quiet --skip_train --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
    for scene in dtu_scenes:
        source = f'{dataset_dir}/{scene}'
        save_dir = f'{output_dir}/{scene}'
        ope = f'CUDA_VISIBLE_DEVICES={device} python eval_render.py --iteration {iteration} -s {source} -m {save_dir} --yaml {yaml_file} --scene {scene}' + common_args

        print(ope)
        os.system(ope)

if not args.skip_metrics:
    ope = f"python scripts/metric_eval.py {output_dir} {30000}"
    os.system(ope)

    ### follow 2dgs mesh extraction and CD calculation.
    # for scene in dtu_scenes:
    #     scan_id = scene[4:]
    #     script_dir = './scripts'
    #     exp_path = f'{output_dir}/{scene}'
    #     ply_file = f"{exp_path}/train/ours_{iteration}/"
    #     string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
    #         f"--input_mesh {exp_path}/train/ours_{iteration}/fuse_post.ply " + \
    #         f"--scan_id {scan_id} --output_dir {output_dir}/tmp/scan{scan_id} " + \
    #         f"--mask_dir {dtu_data} " + \
    #         f"--DTU {dataset_dir}"
        
    #     os.system(string)