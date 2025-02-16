from src.calibration import CameraCalibration
from src.tracking import CameraTracking


# Configuration
GRID_SIZE = (9, 6)
REAL_CELL_SIZE_MM = 16
IMAGE_SIZE = (3024, 4032)
REJECTION_TRESHOLD = 100


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
        tracker.test_image('./tests/', 'test_image.jpg')
        print('Saving camera positions...')
        tracker.plot_camera_position('img')


def main() -> None:
    assigment_1()


if __name__ == '__main__':
    main()
