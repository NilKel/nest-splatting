
import numpy as np
import os
import enum
import types
from typing import List, Mapping, Optional, Text, Tuple, Union
import copy
from PIL import Image
import mediapy as media
from matplotlib import cm
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch


def visualize_volume_rendering(pixel_data):
    # print('pixel_data: ', pixel_data.shape)
    # pixel_data shape : 3*3*N*4
    N = pixel_data.shape[2]
    pixel_data = pixel_data.reshape(-1, N, 4)
    pixel_data = pixel_data[4]

    distances = pixel_data[:, 0]
    G_values = pixel_data[:, 1]
    alpha_values = pixel_data[:, 2]
    T_values = pixel_data[:, 3]

    fig, axs = plt.subplots(3, 1, figsize=(20, 12))
    MAX_N = 20

    mask_valid = (distances > 0.0)
    distances = distances[mask_valid]
    G_values = G_values[mask_valid]
    alpha_values = alpha_values[mask_valid]
    T_values = T_values[mask_valid]
    N = distances.shape[0]
    
    if N == 0:
        return
    
    from utils.render_utils import gsnum_trans_color
    colors = torch.tensor([idx for idx in range(20)])
    colors = gsnum_trans_color(colors, MAX_N = N).permute(1,0).numpy()

    print(f'distances \n {distances}')
    offset = 1e-4

    min_distance = np.floor(np.min(distances.numpy()) * 100) / 100
    max_distance = np.ceil(np.max(distances.numpy()) * 100) / 100
    # print(f'min dis {min_distance} max dis {max_distance}')

    ticks = 0.01
    for ax in axs:
        ax.set_xticks(np.arange(min_distance, max_distance + ticks, ticks))
        ax.set_ylim([0, 1])
        ax.set_xlim([min_distance - ticks, max_distance + ticks]) 
        # print("x ticks:" , np.arange(min_distance, max_distance + ticks, ticks))

    for i in range(N):
        axs[0].bar(distances[i] - (N-i)*offset, alpha_values[i], width=offset, color=colors[i], alpha=0.7, label=f'Point {i+1}')
    axs[0].set_title("Alpha vs Distance")

    for i in range(N):
        axs[1].bar(distances[i]- (N-i)*offset, G_values[i], width=offset, color=colors[i], alpha=0.7, label=f'Point {i+1}')
    axs[1].set_title("G vs Distance")

    for i in range(N):
        axs[2].bar(distances[i]- (N-i)*offset, T_values[i], width=offset, color=colors[i], alpha=0.7, label=f'Point {i+1}')
    axs[2].set_title("T vs Distance")

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.show()

def on_click(event, tensor_data, image):
  x, y = int(event.xdata), int(event.ydata)
  print(f"Clicked at pixel: ({x}, {y})")
  
  H, W, N, _ = tensor_data.shape
  if 0 < x < W-1 and 0 < y < H-1:
      selected_pixels = tensor_data[y-1:y+2, x-1:x+2, :, :]  # 5*5*N*4
      visualize_volume_rendering(selected_pixels)
