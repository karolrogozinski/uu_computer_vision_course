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


def generate_xml() -> None:
    root = ET.Element("opencv_storage")
