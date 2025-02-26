import cv2 as cv
import numpy as np

from utils import extract_frames


def create_background_model(video_path: str, num_frames: int = 15) -> np.ndarray:
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
    kernel = np.ones((kernel_size, kernel_size))

    for _ in range(loops):
        mask = cv.erode(mask, kernel, iterations=iterations)
        mask = cv.dilate(mask, kernel, iterations=iterations)

    return mask


back_path = './data/cam1/background.avi'
fore_path = './data/cam1/video.avi'
back = create_background_model(back_path, 10)
frame = extract_frames(fore_path, 1)[0]

mask = background_substraction(frame, back)
mask = apply_morph_operations(mask, iterations=1, loops=2)
cv.imshow('img', mask)
cv.waitKey(0)
cv.destroyAllWindows()