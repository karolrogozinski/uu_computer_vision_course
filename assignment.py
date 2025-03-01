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
    camera_positions = list()
    cameras = ['cam1', 'cam2', 'cam3', 'cam4']

    for camera in cameras:
        calibration_dict[camera] = read_xml(f"data/{camera}/config.xml")
        rvec = calibration_dict[camera]['rvec']
        tvec = calibration_dict[camera]['tvec']

        R, _ = cv.Rodrigues(rvec)
        R_inv = R.T  # R^{-1} = R^T dla macierzy rotacyjnych
        camera_position = -np.dot(R_inv, tvec).flatten()[[0, 2, 1]]
        camera_position[1] = -camera_position[1]
        camera_positions.append(camera_position)

    print(camera_positions)

    return camera_positions, \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    calibration_dict = dict()
    cameras = ['cam1', 'cam2', 'cam3', 'cam4']

    for camera in cameras:
        calibration_dict[camera] = read_xml(f"data/{camera}/config.xml")

    camera_rotations = list()
    for camera in cameras:
        rvec = calibration_dict[camera]['rvec']

        # Konwersja wektora rotacji do macierzy rotacji
        R, _ = cv.Rodrigues(rvec)
        R_inv = R.T

        R_inv[[1, 2], :] = -R_inv[[1, 2], :]

        R_inv = R_inv[[0, 2, 1], :]  
        R_inv[1, :] = -R_inv[1, :]

        look_at_fix = np.array([
            [0, 0, -1],  
            [0, 1, 0],  
            [1, 0, 0]   
        ])
        R_corrected = np.dot(look_at_fix, R_inv)
        
        glm_matrix = glm.mat4(glm.mat3(*R_corrected.flatten()))
        camera_rotations.append(glm_matrix)

    print(camera_rotations)
    return camera_rotations
