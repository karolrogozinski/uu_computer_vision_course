import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET


def extract_frames(video_path: str, num_frames: int) -> list[np.ndarray]:
    """
    Extracts evenly spaced frames from a video file.

    :param video_path: Path to the video file.
    :param num_frames: Number of frames to extract.
    """
    frames = list()
    capture = cv.VideoCapture(video_path)
    all_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    step = int(all_frames / num_frames)

    frame_idx = 0
    capture.set(1, frame_idx)
    while frame_idx < all_frames:
        _, image = capture.read()
        if frame_idx % step == 0 and image is not None:
            frames.append(image)
        frame_idx += 1
    capture.release()

    return frames


def generate_xml(filename: str, calibration_dict: dict) -> None:
    """
    Saves the calibration dictionary to an XML file.

    :param filename: Path to the XML file.
    :param calibration_dict: Dictionary containing camera parameters.
    """
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)

    fs.write("CameraMatrix", calibration_dict['mtx'])
    fs.write("DistortionCoeffs", calibration_dict['dist'])
    fs.write("RotationVector", calibration_dict['rvec'])
    fs.write("TranslationVector", calibration_dict['tvec'])

    fs.release()


def read_xml(filename: str) -> dict:
    """
    Reads calibration data from an XML file and returns it as a dictionary.

    :param filename: Path to the XML file.
    :return: Dictionary containing camera parameters.
    """
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)

    calibration_dict = {
        "mtx": fs.getNode("CameraMatrix").mat(),
        "dist": fs.getNode("DistortionCoeffs").mat(),
        "rvec": fs.getNode("RotationVector").mat(),
        "tvec": fs.getNode("TranslationVector").mat()
    }

    fs.release()
    return calibration_dict
