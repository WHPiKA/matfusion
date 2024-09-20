from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, UNet2DModel, DDPMScheduler
from matfusion_jax.vis import display_image, display_svbrdf, show_svbrdf
import os
import imageio.v3 as iio
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

device = 'cuda'

model = UNet2DModel.from_pretrained('./checkpoints/flash_v1_diffusers/').to(device)

def halfway_vectors(w, h, camera_pos, light_pos):
    surface = np.indices((h, w)) + 0.5
    surface = np.stack((
        2 * surface[1, :, :] / h - 1,
        -2 * surface[0, :, :] / w + 1,
        np.zeros((h, w), np.float32),
    ), axis=-1)

    wi = light_pos - surface
    wi /= np.linalg.norm(wi, axis=2, keepdims=True)

    wo = camera_pos - surface
    wo /= np.linalg.norm(wo, axis=2, keepdims=True)

    hw = wi + wo
    hw /= np.linalg.norm(hw, axis=2, keepdims=True)

    return hw

for folder in tqdm(os.listdir('/mnt/workspace/data/materialgan')):
    # TODO: pick your own image
    # make sure it is linear RGB with a 2.2 gamma (roughly sRGB without tonemapping)
    if folder.endswith('.sh') or folder.endswith('.py'):
        continue
    image = iio.imread(f'/mnt/workspace/data/materialgan/{folder}/00.png')
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image, 0)
    image = (image / 255.0) ** 2.2
    # display_image(image, 2.2)

    # in addition to the input image itself, the flash finetuning expects the surface's halfway vector

    # accurate FOV can make a pretty big quality difference
    fov = 45

    distance = 1/np.tan(np.deg2rad([fov])/2)
    hw = torch.tensor(
        halfway_vectors(*image.shape[1:3], [0, 0, distance.item()], [0, 0, distance.item()]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).permute(0, 3, 1, 2)

    model_input = torch.tensor(image, device=device, dtype=torch.float32).permute(0, 3, 1, 2) * 2 - 1
    model_input = torch.cat((model_input, hw), 1)

    euler_a_schedule = EulerAncestralDiscreteScheduler(
        beta_schedule='linear',
        prediction_type='v_prediction',
        timestep_spacing="linspace",
    )
    ddim_schedule = DDIMScheduler(
        beta_schedule='linear',
        prediction_type='v_prediction',
        rescale_betas_zero_snr=True,
        clip_sample=False,
        timestep_spacing="linspace",
    )
    schedule = euler_a_schedule
    schedule.set_timesteps(20)
    timestep_mult = model.config.get('timestep_mult', 1/1000)

    diffusion_frames = []

    with torch.no_grad():
        y = torch.randn(1, 10, *model_input.shape[2:], device=device, dtype=torch.float32)
        y = y * schedule.init_noise_sigma

        for t in schedule.timesteps:
            noisy_svbrdf = schedule.scale_model_input(y, t)
            model_output = model(
                torch.cat((noisy_svbrdf, model_input), 1),
                t*timestep_mult,
            ).sample

            step_output = schedule.step(model_output, t, y)
            y = step_output.prev_sample

            svbrdf_est = (step_output.pred_original_sample * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            diffusion_frames.append(svbrdf_est)

    # show_svbrdf(np.concatenate(diffusion_frames), horizontal=True, gamma=2.2)

    svbrdf_img = display_svbrdf(svbrdf_est[0], horizontal=True, format='png', gamma=2.2)
    Path(f'./demo/{folder}_svbrdf.png').write_bytes(svbrdf_img.data) # optional: save the svbrdf to disk
