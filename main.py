from src.calibration import CameraCalibration
from src.tracking import CameraTracking


# Configuration
GRID_SIZE = (9, 6)
REAL_CELL_SIZE_MM = 16
IMAGE_SIZE = (3024, 4032)


def assigment_1() -> None:

    # Offline phase
    calibration = CameraCalibration(
        grid_size=GRID_SIZE, cell_size=REAL_CELL_SIZE_MM)

    calibration.process_images()  
    calibrations = calibration.calibrate_camera(IMAGE_SIZE)

    # Online loops
    for mtx, dist in calibrations:
        tracker = CameraTracking(mtx, dist, calibration.objp,
                                 REAL_CELL_SIZE_MM)
        tracker.track()


def main() -> None:
    assigment_1()


if __name__ == '__main__':
    main()
