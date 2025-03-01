import cv2 as cv
import numpy as np

from utils import extract_frames


def create_background_model(video_path: str, num_frames: int = 15) -> np.ndarray:
    """
    Creates a background model from video.

    :param video_path: Path to a video with a background.
    :param num_frames: Number of frames we want out model to be based on.
    """
    capture = cv.VideoCapture(video_path)
    all_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    num_frames = min(num_frames, all_frames)

    mog2 = cv.createBackgroundSubtractorMOG2(history=num_frames, varThreshold=16, detectShadows=True)

    for _ in range(num_frames):
        ret, frame = capture.read()
        if not ret:
            break
        mog2.apply(frame)
    capture.release()

    return mog2.getBackgroundImage()


def background_substraction(foreground_frame: np.ndarray, background_frame: np.ndarray, \
                            threshold_sat: int = 40, threshold_hue: int = 40, threshold_val: int = 50
                            ) -> np.ndarray:
    """
    Creates a mask of foreground pixels by comparing background model to a frame.

    :param foreground_frame: Frame to extract foreground from.
    :param background_frame: Background model.
    :param threshold_sat: Threshold for saturation.
    :param threshold_hue: Threshold for hue.
    :param threshold_val: Threshold for value.
    """

    foreground_frame = cv.cvtColor(foreground_frame, cv.COLOR_BGR2HSV)
    background_frame = cv.cvtColor(background_frame, cv.COLOR_BGR2HSV)

    # Compute absolute differences in each channel
    diff_hue = cv.absdiff(foreground_frame[:, :, 0], background_frame[:, :, 0])
    diff_sat = cv.absdiff(foreground_frame[:, :, 1], background_frame[:, :, 1])
    diff_val = cv.absdiff(foreground_frame[:, :, 2], background_frame[:, :, 2])

    # Thresholding to create masks
    mask_hue = cv.threshold(diff_hue, threshold_hue, 255, cv.THRESH_BINARY)[1]
    mask_sat = cv.threshold(diff_sat, threshold_sat, 255, cv.THRESH_BINARY)[1]
    mask_val = cv.threshold(diff_val, threshold_val, 255, cv.THRESH_BINARY)[1]

    # Combine the masks (logical OR operation)
    foreground_mask = cv.bitwise_or(mask_hue, mask_sat)
    foreground_mask = cv.bitwise_or(foreground_mask, mask_val)

    return foreground_mask


def apply_morph_operations(mask: np.ndarray, kernel_size: int = 3, iterations: int = 3, loops: int = 1) -> np.ndarray:
    """
    Enhances mask quality by applying morphological operations

    :param mask: Foreground mask.
    :param kernel_size: Size of the kernel used for the morphological operations.
    :param iterations: The number of times the morphological operations are applied.
    :param loops: The number of times to repeat the sequence of erosion followed by dilation.
    
    """
    # TODO: apply more sophisticated enhancers (Blob detection or Graph cuts)
    kernel = np.ones((kernel_size, kernel_size))

    for _ in range(loops):
        mask = cv.erode(mask, kernel, iterations=iterations)
        mask = cv.dilate(mask, kernel, iterations=iterations)

    return mask


def combine_masks(masks: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """
    Combines multiple binary masks using temporal averaging.

    :param masks: List of masks to combine.
    :param threshold: Percentage of frames in which a pixel must be foreground to be considered foreground.
    """
    stacked_masks = np.stack(masks, axis=0)
    avg_mask = np.mean(stacked_masks, axis=0)

    mask = (avg_mask > (threshold * 255)).astype(np.uint8) * 255
    
    return mask


def create_mask(back_path: str, fore_path: str) -> np.ndarray:
    """
    Performs a pipeline of creating a foreground mask by performing background subtraction 
    on multiple frames and then combining them using temporal averaging.

    :param back_path: Path to a video with a background.
    :param fore_path: Path to a video containing the foreground frames to analyze.
    """
    back = create_background_model(back_path, 10)
    frames = extract_frames(fore_path, 100)
    masks = list()
    for frame in frames:
        mask = background_substraction(frame, back)
        mask = apply_morph_operations(mask, iterations=1, loops=2)
        masks.append(mask)
    mask = combine_masks(masks)

    return mask


back_path = './data/cam1/background.avi'
fore_path = './data/cam1/video.avi'
mask = create_mask(back_path, fore_path)
print(mask)
cv.imshow('img', mask)
cv.waitKey(0)
cv.destroyAllWindows()