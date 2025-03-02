import random

import cv2 as cv
import glm
import numpy as np

from src.background import create_mask
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
    """
    Main function to generate voxels
    """

    # Read camera params
    cameras = ['cam1', 'cam2', 'cam3', 'cam4']
    camera_positions, _ = get_cam_positions()
    camera_rotations = get_cam_rotation_matrices()
    camera_params = []

    for i, camera in enumerate(cameras):
        camera_dict = read_xml(f"data/{camera}/config.xml")
        rvec = transform_rvec_to_world(camera_dict['rvec'])
        K = camera_dict['mtx']
        dist = camera_dict['dist']
        mask = create_mask(f"data/{camera}/background.avi", f"data/{camera}/video.avi")

        # Small rvec fix
        R, _ = cv.Rodrigues(rvec)
        R_fix = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        R_corrected = np.dot(R_fix, R)
        tvec_world = np.array(camera_positions[i], dtype=np.float32)

        camera_params.append((R_corrected, tvec_world, K, dist, mask))

    data, colors = [], []

    voxel_maps = dict()
    for i in range(len(cameras)):
        voxel_maps[i] = np.zeros((width, height, depth))

    for x in range(width):
        for y in range(height):
            for z in range(depth):

                voxel = np.array([
                    [int(x * block_size - width / 2),
                     int(y * block_size),
                     int(z * block_size - depth / 2)]
                ], dtype=np.float32)

                visible_cameras = []
                for i, (rvec, _, K, dist, mask) in enumerate(camera_params):
                    # Reconstruction is worse on bigger scale
                    if i not in [1, 2]:
                        continue

                    C = np.array(camera_positions[i])
                    R = np.array(camera_rotations[i])[:3, :3]

                    # Some fixes due to inperfect calibration
                    if i == 1:
                        # C[2:,] += 40
                        R = R @ rotation_matrix_z(-4)
                        R = R @ rotation_matrix_y(-10)
                        R = R @ rotation_matrix_x(5)
                    if i == 2:
                        C[1:,] -= 49
                        C[2:,] += 39
                        R = R @ rotation_matrix_x(0)
                        R = R @ rotation_matrix_y(25)
                        R = R @ rotation_matrix_z(18)

                    transformed_voxel = R @ (voxel.T - C.reshape(3, 1))  
                    if i in [1, 2]:
                        transformed_voxel = transformed_voxel[[2, 1, 0], :]
                        if i == 1:
                            transformed_voxel[2, :] *= -1
                            transformed_voxel[0, :] *= -1
                        
                    else:
                        transformed_voxel = transformed_voxel[[0, 1, 2], :]
                        transformed_voxel[0, :] *= -1
                        if i == 3:
                            transformed_voxel[0, :] *= -1
                            transformed_voxel[2, :] *= -1

                    # Manual 3D to 2D
                    # (projectPoints doesnt work with modified camera params)
                    P_homogeneous = K @ transformed_voxel
                    image_points = P_homogeneous[0] / P_homogeneous[2], \
                        P_homogeneous[1] / P_homogeneous[2]
                    x_proj, y_proj = int(image_points[0]), int(image_points[1])

                    # Mask check
                    if 0 <= x_proj < mask.shape[1] and 0 <= y_proj < mask.shape[0]:
                        if mask[y_proj, x_proj] > 0:
                            visible_cameras.append(i)
                            voxel_maps[i][int(voxel[0, 0]), int(voxel[0, 1]), int(voxel[0, 2])] = 1

                if visible_cameras:
                    if visible_cameras != [1, 2]:
                        continue
                    voxel_color = (1.0, 0.7529, 0.7961)

                    data.append(voxel.flatten())
                    colors.append(tuple(voxel_color))

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
        rvec = transform_rvec_to_world(rvec)
        R_corrected, _ = cv.Rodrigues(rvec)

        glm_matrix = glm.mat4(glm.mat3(*R_corrected.flatten()))
        camera_rotations.append(glm_matrix)

    print(camera_rotations)
    return camera_rotations


"""
Some small help functionalities
"""


def rotation_matrix_y(angle_degrees: int) -> np.ndarray:
    angle = np.radians(angle_degrees)
    return np.array([
        [np.cos(angle),  0, np.sin(angle)],
        [0,              1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])


def rotation_matrix_x(angle_degrees: int) -> np.ndarray:
    angle = np.radians(angle_degrees)
    return np.array([
        [1, 0,              0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)]
    ])


def rotation_matrix_z(angle_degrees: int) -> np.ndarray:
    angle = np.radians(angle_degrees)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])


def transform_rvec_to_world(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv.Rodrigues(rvec)
    R_inv = R.T

    R_inv[[1, 2], :] = -R_inv[[1, 2], :]
    R_inv = R_inv[[0, 2, 1], :]  
    R_inv[1, :] = -R_inv[1, :]

    correction_matrix = np.array([
        [0, 0, -1],  
        [0, 1, 0],  
        [1, 0, 0]   
    ])
    R_corrected = np.dot(correction_matrix, R_inv)
    rvec_corrected, _ = cv.Rodrigues(R_corrected)
    return rvec_corrected
