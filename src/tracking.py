import cv2 as cv
import numpy as np


class CameraTracking:
    """
    Handles real-time chessboard object tracking using a calibrated camera.
    """

    def __init__(self, mtx: np.ndarray, dist: np.ndarray,
                 objp: np.ndarray, cell_size: int):
        """
        Initializes the CameraTracking instance.

        :param mtx: Camera matrix.
        :param dist: Distortion coefficients.
        :param objp: 3D object points for the chessboard.
        :param cell_size: Size of a single chessboard cell in some units.
        """
        self.mtx = mtx
        self.dist = dist
        self.objp = objp
        self.cell_size = cell_size

    def test_image(self, dir, image):
        frame = cv.imread(f'./img/{image}')
        self._process_frame(frame)
        cv.imwrite(f'{dir}{image}', frame)

    def track(self) -> None:
        """
        Starts the real-time tracking process using a webcam.
        """
        cap = cv.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                self._process_frame(frame)
            cv.imshow('Live Tracking', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):  # q for breaking loop
                break
        cap.release()
        cv.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray) -> None:
        """
        Processes a frame to detect a chessboard and estimate pose.

        Args:
            frame (np.ndarray): The input image frame.
        """
        self.gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(self.gray, (9, 6), None)
        if not found:
            return

        corners2 = cv.cornerSubPix(
            self.gray, corners, (11, 11), (-1, -1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        _, rvec, tvec = cv.solvePnP(self.objp, corners2, self.mtx, self.dist)
        self._draw_objects(frame, rvec, tvec)

    def _draw_objects(
        self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
    ) -> None:
        """
        Draws 3D objects (axes and cube) on the given frame.

        Args:
            frame (np.ndarray): The input image frame.
            rvec (np.ndarray): The rotation vector.
            tvec (np.ndarray): The translation vector.
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
                       (100, 100, 100), thickness=10)

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

        Args:
            img (np.ndarray): The image on which to draw the axes.
            origin (np.ndarray): The origin point of the axes.
            imgpts (np.ndarray): The projected points representing the axes.
        """
        imgpts = np.int32(imgpts).reshape(-1, 2)
        origin = np.int32(origin).reshape(2)

        cv.line(img, origin, tuple(imgpts[1]), (0, 0, 255), 15)
        cv.line(img, origin, tuple(imgpts[2]), (0, 255, 0), 15)
        cv.line(img, origin, tuple(imgpts[3]), (255, 0, 0), 15)

    @staticmethod
    def draw_cube(
        img: np.ndarray, imgpts: np.ndarray,
        color: tuple[int, int, int], thickness: int = 5
    ) -> None:
        """
        Draws a cube using projected 2D points.

        Args:
            img (np.ndarray): The image on which to draw the cube.
            imgpts (np.ndarray): The projected 2D points of the cube.
            color (Tuple[int, int, int]): The color of the cube edges (B, G, R).
            thickness (int, optional): The thickness of the edges. Defaults to 5.
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

        Args:
            tvec (np.ndarray): The translation vector.
            rvec (np.ndarray): The rotation vector.

        Returns:
            Tuple[int, int, int]: The computed BGR color.
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

        # TODO fix this color (I think its not correctly converted from HSV tp BGR)
        bgr_color = cv.cvtColor(np.uint8([[[H, S, V]]]),
                                cv.COLOR_HSV2BGR)[0][0]

        return tuple(map(int, bgr_color))
