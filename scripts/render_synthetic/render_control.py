import sys

print('Python', sys.version)

import bpy

scene = bpy.data.scenes['Scene']
prefs = bpy.context.preferences.addons['cycles'].preferences

all_devices = prefs.get_devices()
if all_devices is None:
    print('No Devices')
else:
    print('Devices', ', '.join(dev.name for devs in prefs.get_devices() for dev in devs))

scene.cycles.device = 'CPU'
prefs.compute_device_type = 'CUDA'

output_file_image = scene.node_tree.nodes['output_image']
output_file_position = scene.node_tree.nodes['output_position']

import numpy as np
from glob import glob
import random
import re
import json
import math

for input_info in sys.stdin:
    input_info = json.loads(input_info)
    name = input_info['name']
    output_dir = input_info['output_dir']
    print(f"== {name} ==", flush=True)
    if scene.cycles.device == 'CPU' and input_info['use_cuda']:
        scene.cycles.device = 'GPU'

    output_file_image.base_path = output_dir
    output_file_position.base_path = output_dir

    bpy.data.images['height'].filepath = input_info['height']
    bpy.data.images['diffuse'].filepath = input_info['diffuse']
    bpy.data.images['diffuse'].colorspace_settings.name = input_info['colorspace']
    bpy.data.images['specular'].filepath = input_info['specular']
    bpy.data.images['specular'].colorspace_settings.name = input_info['colorspace']
    bpy.data.images['roughness'].filepath = input_info['roughness']
    bpy.data.images['normals'].filepath = input_info['normals']
    env = input_info['world']
    bpy.data.images['env'].filepath = env

    env_energy = 1.0
    flash_energy = 50.0
    flash = bpy.data.objects['flash']
    target = bpy.data.objects['target']
    camera = bpy.data.objects['camera']
    bounds = bpy.data.objects['plane'].bound_box

    if input_info.get('flash_offset') is not None:
        flash_offset = input_info['flash_offset']
    else:
        flash_offset = np.random.normal(0.0, 0.05, 2).tolist()
    flash.location.xy = flash_offset
    if input_info.get('world_rotation') is not None:
        world_rotation = input_info['world_rotation']
    else:
        world_rotation = np.random.uniform(0.0, 360.0, 1).item()
    target['world_rotation'] = world_rotation

    views = {}
    for variation in ('a', 'b'):
        if input_info.get('target_location') is not None:
            target_location = input_info['target_location']
        else:
            target_location = np.random.uniform(-0.1, 0.1, 2).tolist()
        target.location.xy = target_location
        if input_info.get('target_rotation') is not None:
            target_rotation = input_info['target_rotation']
        else:
            target_rotation = [1, *np.random.uniform(-0.02, 0.02, 3)]
        target.rotation_quaternion = target_rotation
        if input_info.get('flash_distance') is not None:
            camera.location.z = input_info['flash_distance']
            camera.data.lens = 1000 * input_info['flash_distance']
            flash.location.z = input_info['flash_distance']
        else:
            camera.location.z = 2.0
            camera.data.lens = 2000
            flash.location.z = 2.0
        views[variation] = {
            'target': target_location,
            'rotation': target_rotation,
            'distance': camera.location.z,
            'fov': math.degrees(camera.data.angle),
        }

        target['world_energy'] = env_energy
        flash.data.energy = 0
        output_file_position.mute = False
        output_file_position.file_slots[0].path = f'{name}_{variation}_position_'
        output_file_image.file_slots[0].path = f'{name}_{variation}_envio_'
        bpy.ops.render.render()

        target['world_energy'] = 0
        flash.data.energy = flash_energy
        output_file_position.mute = True
        output_file_image.file_slots[0].path = f'{name}_{variation}_flash_'
        bpy.ops.render.render()

    with open(f'{output_dir}/{name}.json', 'w') as meta:
        meta.write(json.dumps({
            'flash_offset': flash_offset,
            'world': env,
            'world_rotation': world_rotation,
            'bounds': {
                'x': (min(p[0] for p in bounds), max(p[0] for p in bounds)),
                'y': (min(p[1] for p in bounds), max(p[1] for p in bounds)),
                'z': (min(p[2] for p in bounds), max(p[2] for p in bounds)),
            },
            'views': views,
        }))

    print('READY', flush=True)
