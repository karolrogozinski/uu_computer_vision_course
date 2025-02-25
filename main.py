import datetime

import cv2 as cv

from src.calibration import CameraCalibration
from src.tracking import CameraTracking
from src.utils import extract_frames


# Configuration
GRID_SIZE = (6, 8)
REAL_CELL_SIZE_MM = 115
IMAGE_SIZE = (486, 644)
REJECTION_TRESHOLD = 100
FRAME_SAMPLING = 15


def assigment_1() -> None:

    # Offline phase
    calibration = CameraCalibration(
        grid_size=GRID_SIZE, cell_size=REAL_CELL_SIZE_MM)

    calibration.process_images()
    calibrations = calibration.calibrate_camera(
        IMAGE_SIZE, REJECTION_TRESHOLD)

    # Online loops
    for mtx, dist in calibrations:
        tracker = CameraTracking(mtx, dist, calibration.objp,
                                 REAL_CELL_SIZE_MM)
        print('--------------------')
        print('Starting camera... (press "q" to end live view)')
        tracker.track()
        print('Saving test image...')
        tracker.test_image('./tests/test_image.jpg')
        print('Saving camera positions...')
        tracker.plot_camera_position('img')


def assigment_2() -> None:

    # Cameras calibration
    calibration = CameraCalibration(
        grid_size=GRID_SIZE, cell_size=REAL_CELL_SIZE_MM
    )

    cameras = ['cam1', 'cam2', 'cam3', 'cam4']
    calibration_dict = dict()

    for camera in cameras:
        calibration_dict[camera] = dict()

        # intrinsics

        video_path = f'./data/{camera}/intrinsics.avi'
        frames = extract_frames(video_path, FRAME_SAMPLING)

        calibration.process_frames_list(frames)
        calibrations = calibration.calibrate_camera(
            IMAGE_SIZE, REJECTION_TRESHOLD)

        calibration_dict[camera]['mtx'] = calibrations[0][0]
        calibration_dict[camera]['dist'] = calibrations[0][1]

        # checkerboard

        video_path = f'./data/{camera}/checkerboard.avi'
        frames = extract_frames(video_path, 1)
        calibration.process_frames_list(frames)

        objp = calibration.objpoints[0]
        corners = calibration.imgpoints[0]
        mtx, dist = calibrations[0]

        tracker = CameraTracking(mtx, dist, calibration.objp,
                                 REAL_CELL_SIZE_MM, GRID_SIZE)
        _, rvec, tvec = cv.solvePnP(objp, corners, mtx, dist)
        tracker.draw_objects(frames[0], rvec, tvec)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_image_{timestamp}.png"

        cv.imwrite(f'tests/{filename}', frames[0])

        calibration_dict[camera]['rvec'] = rvec
        calibration_dict[camera]['dist'] = tvec


def main() -> None:
    assigment_2()


if __name__ == '__main__':
    main()
