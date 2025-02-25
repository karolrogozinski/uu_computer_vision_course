import datetime
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class CameraTracking:
    """
    Handles real-time chessboard object tracking using a calibrated camera.
    """

    def __init__(self, mtx: np.ndarray, dist: np.ndarray,
                 objp: np.ndarray, cell_size: int, grid_size: tuple[int] = (9, 6)):
        """
        Initializes the CameraTracking instance.

        :param mtx: Camera matrix.
        :param dist: Distortion coefficients.
        :param objp: 3D object points for the chessboard.
        :param cell_size: Size of a single chessboard cell in some units.
        :param grid_size: Number of rows and columns in a checkerboard.
        """
        self.rows = grid_size[0]
        self.cols = grid_size[1]

        self.mtx = mtx
        self.dist = dist
        self.objp = objp
        self.cell_size = cell_size

    def test_image(self, img: str | np.ndarray, source: str = 'file'):
        """
        Draw a polygon that covers the top side of the cube on the given img

        :param img: Filename of the image
        :param source: Source of the img - file | variable.
        """
        if source == 'file':
            img = cv.imread(img)

        rvec, tvec = self._process_frame(img)
        self.draw_objects(img, rvec, tvec)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_image_{timestamp}.png"

        cv.imwrite(f'{dir}{filename}', img)

    def track(self) -> None:
        """
        Starts the real-time tracking process using a webcam.
        """
        cap = cv.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                try:
                    rvec, tvec = self._process_frame(frame)
                    self._draw_objects(frame, rvec, tvec)
                except TypeError:  # corners not found
                    pass
            cv.imshow('Live Tracking', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):  # q for breaking loop
                break
        cap.release()
        cv.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray) -> None:
        """
        Processes a frame to detect a chessboard and estimate pose.

        :param frame: The input image frame.
        """
        self.gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(
            self.gray, (self.rows, self.cols), None
        )
        if not found:
            return

        corners2 = cv.cornerSubPix(
            self.gray, corners, (11, 11), (-1, -1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        _, rvec, tvec = cv.solvePnP(self.objp, corners2, self.mtx, self.dist)
        return rvec, tvec

    def draw_objects(
        self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
    ) -> None:
        """
        Draws 3D objects (axes and cube) on the given frame.

        :param frame: The input image frame.
        :param rvec: The rotation vector.
        :param tvec: The translation vector.
        """

        # Axes
        axis_points = np.float32(
            [[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]
        ) * self.cell_size
        axis_imgpts, _ = cv.projectPoints(axis_points, rvec, tvec,
                                          self.mtx, self.dist)
        self.draw_axes(frame, tuple(axis_imgpts[0].ravel()),
                       np.int32(axis_imgpts).reshape(-1, 2))

        # Cube
        cube_points = np.float32([
            [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
            [0, 0, -2], [2, 0, -2], [2, 2, -2], [0, 2, -2]
        ]) * self.cell_size
        cube_imgpts, _ = cv.projectPoints(cube_points, rvec, tvec,
                                          self.mtx, self.dist)
        self.draw_cube(frame, np.int32(cube_imgpts).reshape(-1, 2),
                       (200, 200, 200), thickness=5)

        # Polygon
        top_color = self.get_dynamic_color(tvec, rvec)
        cv.fillConvexPoly(
            frame, np.array([cube_imgpts[4], cube_imgpts[5], cube_imgpts[6],
                             cube_imgpts[7]], dtype=np.int32), top_color
        )

    @staticmethod
    def draw_axes(img: np.ndarray, origin: np.ndarray, imgpts: np.ndarray
                  ) -> None:
        """
        Draws the XYZ coordinate axes from the origin.

        :param img: The image on which to draw the axes.
        :param origin: The origin point of the axes.
        :param imgpts: The projected points representing the axes.
        """
        imgpts = np.int32(imgpts).reshape(-1, 2)
        origin = np.int32(origin).reshape(2)

        cv.line(img, origin, tuple(imgpts[1]), (0, 0, 255), 5)
        cv.line(img, origin, tuple(imgpts[2]), (0, 255, 0), 5)
        cv.line(img, origin, tuple(imgpts[3]), (255, 0, 0), 5)

    @staticmethod
    def draw_cube(
        img: np.ndarray, imgpts: np.ndarray,
        color: tuple[int, int, int], thickness: int = 5
    ) -> None:
        """
        Draws a cube using projected 2D points.

        :param img: The image on which to draw the cube.
        :param imgpts : The projected 2D points of the cube.
        :param color: The color of the cube edges (B, G, R).
        :param thickness: The thickness of the edges. Defaults to 5.
        """
        # Base edges
        for i, j in zip(range(4), [1, 2, 3, 0]):
            cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, thickness)
        # Top edges
        for i, j in zip(range(4, 8), [5, 6, 7, 4]):
            cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, thickness)
        # Vertical edges
        for i, j in zip(range(4), range(4, 8)):
            cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, thickness)

    @staticmethod
    def get_dynamic_color(tvec: np.ndarray, rvec: np.ndarray
                          ) -> tuple[int, int, int]:
        """
        Computes the top face color based on distance and orientation.

        :param tvec: The translation vector.
        :param rvec: The rotation vector.

        :return: The computed BGR color.
        """
        # Value based on the euclidean distance from the camera
        distance = np.linalg.norm(tvec)
        V = int(max(0, 255 * (1 - min(distance / 4000, 1))))

        # Saturation based on the camera angle
        R, _ = cv.Rodrigues(rvec)
        top_normal = R[:, 2]
        camera_direction = np.array([0, 0, 1])
        angle = np.arccos(np.dot(top_normal, camera_direction)) * (180 / np.pi)
        S = int(max(0, 255 * (1 - min(angle / 45, 1))))

        # Hue based on the rotation of the camera
        R, _ = cv.Rodrigues(rvec)
        chessboard_x_axis = R[:, 0]
        camera_x_axis = np.array([1, 0, 0])
        dot_product = np.dot(chessboard_x_axis, camera_x_axis)
        yaw_angle = np.arccos(dot_product) * (180 / np.pi)
        H = int(round((yaw_angle % 360)))

        bgr_color = cv.cvtColor(np.uint8([[[H, S, V]]]),
                                cv.COLOR_HSV2BGR)[0][0]

        return tuple(map(int, bgr_color))

    def plot_camera_position(self, img_dir: str) -> None:
        """
        Plots the estimated camera positions and orientations in a 3D space.

        :param img_dir: Directory containing images used for camera pose estimation.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = self.objp[:, 0], self.objp[:, 1], self.objp[:, 2]
        ax.scatter(x, y, z, c='black', marker='x', alpha=0.5, label='Checkerboard')

        for fname in os.listdir(img_dir):
            if fname in ('test_image.jpg', '.gitignore'):
                continue
            img_path = os.path.join(img_dir, fname)

            frame = cv.imread(img_path)
            try:
                rvec, tvec = self._process_frame(frame)
            except TypeError:  # corners not found
                continue

            R, _ = cv.Rodrigues(rvec)
            camera_position = -R.T @ tvec

            x, y, z = camera_position.flatten()
            ax.text(x, y, z, fname[-8:-4], color='black', fontsize=8)
            ax.scatter(x, y, z, c='purple', marker='o', label='Camera Position')
            ax.quiver(x, y, z, R[0, 2], -R[1, 2], R[2, 2], length=60,
                      color='purple', alpha=.25)

        ax.invert_zaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Positions')

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./fig/camera_positions_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
