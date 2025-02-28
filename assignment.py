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

    camera_positions = [calibration_dict[c]['tvec'].reshape(-1, 3).tolist()[0]
                        for c in cameras]
    camera_positions = scale_camera_positions(camera_positions)

    return camera_positions, \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    calibration_dict = dict()
    cameras = ['cam1', 'cam2', 'cam3', 'cam4']

    for camera in cameras:
        calibration_dict[camera] = read_xml(f"data/{camera}/config.xml")

    rvecs = [calibration_dict[c]['rvec'].reshape(-1, 3)
             for c in cameras]
    tvecs = [calibration_dict[c]['tvec'].reshape(-1, 3)
             for c in cameras]
    print(rvecs)
    print(tvecs)
    cam_rotations = []

    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv.Rodrigues(rvec)

        print(R.shape, tvec.shape)
        camera_rotation = -R.T @ tvec.T
        print(camera_rotation)

        glm_matrix = glm.mat4(1.0)
        for i in range(3):
            for j in range(3):
                glm_matrix[i][j] = camera_rotation[i, j]

        cam_rotations.append(glm_matrix)

    return cam_rotations

    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
