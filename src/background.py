import cv2 as cv
import numpy as np


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
                            threshold_sat: int, threshold_hue: int, threshold_val: int
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


path = './data/cam1/background.avi'
back = create_background_model(path, 10)
frame = 
cv.imshow('img', back)
cv.waitKey(0)
cv.destroyAllWindows()