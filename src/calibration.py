import os

import cv2 as cv
import numpy as np


class CameraCalibration:
    """
    Handles camera calibration using chessboard pattern detection.
    """

    def __init__(self, grid_size: tuple[int, int],
                 cell_size: int, sort_corners: bool = True) -> None:
        """
        Initializes the CameraCalibration instance.

        :param grid_size: Number of inner corners per of a chessboard.
        :param cell_size: Size of a single chessboard cell in some units.
        :param sort_corners: Flag to sort corners in order: (top-left, top-right,
            bottom-left, bottom-right) during manual phase
        """
        self.criteria = (
            cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.objp = np.zeros((grid_size[1] * grid_size[0], 3), np.float32)
        self.objp[:, :2] = np.mgrid[
            0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * cell_size
        self.grid_size = grid_size
        self.sort_corners = sort_corners

        self.reset_points()

    def reset_points(self) -> None:
        """
        Set points to empty lists
        """
        self.objpoints: list[np.ndarray] = []
        self.imgpoints: list[np.ndarray] = []
        self.detected_automatically: list[bool] = []

    def process_images(self, img_dir: str = 'img') -> None:
        """
        Processes all images in the specified directory to detect the corners.

        :param img_dir: Directory containing calibration images.
        """
        for fname in os.listdir(img_dir):
            if fname in ('test_image.jpg', '.gitignore'):
                continue
            img_path = os.path.join(img_dir, fname)
            self._process_single_image(img_path)
        cv.destroyAllWindows()

    def process_frames_list(self, frames: list[np.ndarray]) -> None:
        """
        Processes all frmaes in the given list to detect the corners.

        :param frames: List containing calibration frames.
        """
        for frame in frames:
            self._process_single_image(frame, source='variable')
        cv.destroyAllWindows()

    def _process_single_image(self, img: str | np.ndarray, source: str = 'file') -> None:
        """
        Processes a single image for chessboard corner detection.

        :param img_path: Path to the image file or loaded frame is source is not file.
        :param source: Source of the img - file | variable.
        """
        if source == 'file':
            img = cv.imread(img)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, self.grid_size, None)

        if not ret:#True: #not ret:
            corners = self._manual_corner_selection(img)
        if corners is not None:
            corners2 = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self.criteria)

            self.objpoints.append(self.objp)
            self.imgpoints.append(corners2)
            self.detected_automatically.append(ret)

            cv.drawChessboardCorners(img, self.grid_size, corners2, True)
            cv.imshow('img', img)
            cv.waitKey(2000)  # Display time - can be adjusted

    def _manual_corner_selection(self, img: np.ndarray) -> np.ndarray:
        """
        Allows the user to manually select chessboard corners
        if automatic detection fails.

        :param img: Image where manual selection is performed.
        :return: Manually selected corner coordinates.
        """
        collector = ManualGrid(*self.grid_size, self.sort_corners)
        cv.imshow('img', img)
        cv.setMouseCallback('img', collector.click_event)
        while True:
            key = cv.waitKey(0)
            if key == 13:  # Enter key (confirm selection)
                break
            elif key == ord('s'):  # "s" key (cancel selection)
                cv.destroyAllWindows()
                print("Skipped image")
                return None
        collector.interpolate()
        return collector.grid

    def calibrate_camera(self, gray_shape: tuple[int, int], rejection_th: float
                         ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Calibrates the camera using the detected chessboard corners.

        :param gray_shape: Shape of the grayscale image used for calibration.
        :return: List of tuples containing camera matrix and distortion coeffs.
        """
        objpoints = np.array(self.objpoints)
        imgpoints = np.array(self.imgpoints)
        idx_sets = self._get_index_sets()

        calibrations = []
        for idxs in idx_sets:
            _, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                objpoints[idxs], imgpoints[idxs], gray_shape, None, None)

            rejected_indices = self.get_bad_quality_indices(
                idxs, tvecs, rvecs, mtx, dist, rejection_th)

            new_idxs = [idx for idx in idxs if idx not in rejected_indices]
            idxs = new_idxs if len(new_idxs) >= 5 else idxs

            _, mtx, dist, _, _ = cv.calibrateCamera(
                objpoints[idxs], imgpoints[idxs], gray_shape, None, None)

            calibrations.append((mtx, dist))
        return calibrations

    def get_bad_quality_indices(self, indices: list[list[int]], tvecs: np.ndarray,
                                rvecs: np.ndarray, mtx: np.ndarray, dist: np.ndarray,
                                rejection_th: float) -> list[int]:
        """
        Identifies calibration images with high reprojection error.

        :param indices: List of index sets representing images used in
            calibration.
        :param tvecs: Translation vectors obtained from calibration.
        :param rvecs: Rotation vectors obtained from calibration.
        :param mtx: Camera matrix.
        :param dist: Distortion coefficients.
        :param rejection_th: Threshold for rejecting images based on
            reprojection error.
        :return: List of indices of images with reprojection error exceeding
            the threshold.
        """
        errors = []
        for i, idx in enumerate(indices):
            reprojected, _ = cv.projectPoints(
                np.array(self.objpoints)[idx], rvecs[i], tvecs[i], mtx, dist)
            mean_error = np.mean(np.linalg.norm(
                np.array(self.imgpoints)[i] - reprojected, axis=2))
            errors.append((mean_error, idx))

        rejected_indices = [idx for error, idx in errors if error > rejection_th]
        return rejected_indices

    def _get_index_sets(self) -> list[list[int]]:
        """
        Generates different index sets for each calibration run.

        :return: List of index sets.
        """
        # run 1: all images
        idxs_1 = list(range(len(self.detected_automatically)))
        # run 2: first 10 automatically detected images
        idxs_2 = [i for i, val in
                  enumerate(self.detected_automatically) if val][:10]
        # run 3: half of run 2
        idxs_3 = idxs_2[:5]
        return [idxs_1, idxs_2, idxs_3]


class ManualGrid:
    """
    A class to generate a manual grid based on user-clicked points
    and interpolate grid points between them.
    """

    def __init__(self, rows: int, cols: int, sort_corners: bool = True) -> None:
        """
        Initializes the ManualGrid with given rows and columns.

        :param rows: Number of rows in the grid, default is 9.
        :param cols: Number of columns in the grid, default is 6.
        :param sort_corners: Flag to sort corners in order: (top-left, top-right,
            bottom-left, bottom-right) during manual phase
        """
        self.grid: np.ndarray = np.zeros([rows * cols, 1, 2], dtype=np.float32)
        self.points: list[list[int]] = []
        self.rows: int = rows
        self.cols: int = cols
        self.sort_corners = sort_corners

    def click_event(self, event: int, x: int, y: int,
                    flags: int, param) -> None:
        """
        Handles mouse click events to capture points.

        :param event: OpenCV event type.
        :param x: X-coordinate of the click.
        :param y: Y-coordinate of the click.
        :param flags: Any relevant flags passed by OpenCV.
        :param param: Additional parameters.
        """
        if event == cv.EVENT_LBUTTONDOWN:
            print('Clicked corner', (x, y))
            self.points.append([x, y])

    def interpolate(self) -> None:
        """
        Interpolates a grid of points based on the user-provided corner points.
        Outputs the grid column by column instead of row by row.
        """
        if self.sort_corners:
            top_left, top_right, bot_left, bot_right = self.sorted_corners()
        else:
            ...
            top_left, top_right, bot_left, bot_right = self.points[-4:]
            # TODO load last 4 corners in given order

        top_row = [
            self.interpolate_pair(top_left, top_right, i / (self.cols - 1))
            for i in range(self.cols)
        ]
        bot_row = [
            self.interpolate_pair(bot_left, bot_right, i / (self.cols - 1))
            for i in range(self.cols)
        ]

        grid_points: list[tuple[float, float]] = []
        for col in range(self.cols):  # Iterujemy po kolumnach
            col_points = [
                self.interpolate_pair(top_row[col], bot_row[col], row / (self.rows - 1))
                for row in range(self.rows)
            ]
            grid_points.extend(col_points)

        self.grid = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)

    @staticmethod
    def interpolate_pair(point1: tuple[float, float],
                         point2: tuple[float, float], proportion: float
                         ) -> tuple[float, float]:
        """
        Linearly interpolates between two points.

        :param point1: First point (x, y).
        :param point2: Second point (x, y).
        :param proportion: Interpolation proportion (0 to 1).
        :return: Interpolated point (x, y).
        """
        x = point1[0] + (point2[0] - point1[0]) * proportion
        y = point1[1] + (point2[1] - point1[1]) * proportion
        return x, y

    def sorted_corners(self) -> tuple[tuple[int, int], tuple[int, int],
                                      tuple[int, int], tuple[int, int]]:
        """
        Sorts the clicked points to identify the four corner points.

        :return: Four corner points in order (top-left, top-right,
        bottom-left, bottom-right).
        """
        points = sorted(self.points, key=lambda p: p[1])
        bot_points = sorted(points[:2], key=lambda p: p[0])
        top_points = sorted(points[-2:], key=lambda p: p[0])
        return top_points[0], top_points[1], bot_points[0], bot_points[1]
