import sys
import os

import cv2 as cv
import glm
import numpy as np
import random

from src.utils import read_xml


block_size = 1.0
WORLD_WIDTH = 128
WORLD_HEIGHT = 64
WORLD_DEPTH = 128


def scale_camera_positions(camera_positions):
    positions = np.array(camera_positions)

    positions[:, 0] = (positions[:, 0] / np.max(np.abs(positions[:, 0]))) * (WORLD_WIDTH / 2)
    positions[:, 1] = (positions[:, 1] / np.max(np.abs(positions[:, 1]))) * WORLD_HEIGHT
    positions[:, 2] = (positions[:, 2] / np.max(np.abs(positions[:, 2]))) * (WORLD_DEPTH / 2)

    return positions.tolist()


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data, colors = [], []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
                    colors.append([x / width, z / depth, y / height])
    return data, colors


def get_cam_positions():
    calibration_dict = dict()
    cameras = ['cam1', 'cam2', 'cam3', 'cam4']

    for camera in cameras:
        calibration_dict[camera] = read_xml(f"data/{camera}/config.xml")
    indices = np.array([0, 2, 1])
    camera_positions = [calibration_dict[c]['tvec'].flatten()[indices] / 50
                        for c in cameras]
    
    # camera_positions = scale_camera_positions(camera_positions)
    # camera_positions = camera_positions / 100


    return camera_positions, \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    calibration_dict = dict()
    cameras = ['cam1', 'cam2', 'cam3', 'cam4']

    for camera in cameras:
        calibration_dict[camera] = read_xml(f"data/{camera}/config.xml")

    rotations = list()
    for c in cameras:
        rvecs = calibration_dict[c]['rvec']
        rvecs, _ = cv.Rodrigues(rvecs)
        # rvecs = rvecs.tolist()
        rotations.append(rvecs)

    glm_matrices = list()
    for rot in rotations:
        glm_matrices.append(glm.mat4(glm.mat3(*rot.flatten())))
    return glm_matrices
