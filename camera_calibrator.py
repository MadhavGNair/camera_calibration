import glob
import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

TILE_SIZE = 16  # size of the chessboard square in mm


class CameraCalibrator:
    def __init__(self, images_root, chessboard_size, square_size):
        self.width, self.height = chessboard_size
        self.square_size = square_size
        self.image_root = images_root
        self.image_paths = glob.glob(images_root + "/*.png")
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.obj_points = np.zeros((self.width * self.height, 3), np.float32)
        self.obj_points[:, :2] = (
            np.mgrid[0 : self.height, 0 : self.width].T.reshape(-1, 2)
            * self.square_size
        )
        self.global_obj_points = []
        self.global_img_points = []

    def __get_mouse_click(self, event, x, y, flags, param):
        corners = param["global_corners"]
        win_title = param["win_title"]
        image = param["image"]
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(win_title, image)
            if len(corners) >= 4:
                cv2.destroyWindow(win_title)
        return corners

    def __select_global_corners(self, image):
        win_title = (
            "Select Corners in order top-left, top-right, bottom-right, bottom-left"
        )
        global_corners = []
        img = image.copy()

        params = {
            "image": img,
            "global_corners": global_corners,
            "win_title": win_title,
        }

        cv2.namedWindow(win_title)
        cv2.setMouseCallback(win_title, self.__get_mouse_click, params)
        cv2.imshow(win_title, image)
        cv2.waitKey(0)
        return global_corners

    def __compute_local_corners(self, image):
        global_corners = self.__select_global_corners(image)
        tl, tr, br, bl = global_corners

        corners = np.float32([tl, tr, br, bl])
        
        # compute the orthogonal rectangle size (when viewed straight ahead)
        ort_width = (self.width - 1) * self.square_size
        ort_height = (self.height - 1) * self.square_size
        
        # define the orthogonal rectangle corners
        ort_corners = np.float32([
            [0, 0],
            [ort_width, 0],
            [ort_width, ort_height],
            [0, ort_height]
        ])
        
        # CHOICE TASK 3: calculate perspective transform matrix (P) that converts the physical 
        # rectangle (straight) to the manually provided rectangle (tilted)
        perspective_matrix = cv2.getPerspectiveTransform(ort_corners, corners)
        
        # compute the local corners
        local_corners = np.zeros((self.width * self.height, 1, 2), np.float32)
        for i in range(self.width):  # Iterate columns
            for j in range(self.height):  # Iterate rows
                x = i * self.square_size
                y = j * self.square_size

                # [x' y' w'] = P * [x y 1]
                point = np.array([x, y, 1.0])
                trans_point = perspective_matrix.dot(point)
                # normalize the homogeneous coordinates
                trans_point = trans_point / trans_point[2]
                
                # Calculate the index based on the cv2 ordering
                index = (self.height - 1 - j) + i * self.height
                
                local_corners[index, 0] = trans_point[:2]
        return local_corners

    def __save_calibration_data(self, cam_matrix, coeffs, rvecs, tvecs, filepath):
        calibration_data_json = {
            "camera_matrix": cam_matrix.tolist(),
            "dist_coeffs": coeffs.tolist(),
            "rvecs": [rvec.tolist() for rvec in rvecs],
            "tvecs": [tvec.tolist() for tvec in tvecs],
        }
        with open(filepath + ".json", "w") as f:
            json.dump(calibration_data_json, f, indent=4)
        print(f"Calibration data saved to {filepath}.json")

    def __load_calibration_data(self, filepath):
        with open(filepath, "r") as f:
            calibration_data = json.load(f)
        return (
            np.array(calibration_data["camera_matrix"]),
            np.array(calibration_data["dist_coeffs"]),
            np.array(calibration_data["rvecs"]),
            np.array(calibration_data["tvecs"]),
        )

    def draw_axes_on_chessboard(self, image_path, params_path, save=False, save_root='./output'):
        """
        Function to draw X, Y, Z axes on the chessboard in the image based on estimated camera parameters.
        :param image_path: Path to the image of the chessboard
        :param params_path: Path to the calibration parameters file
        :param save: Boolean flag to save the image with axes drawn
        :return: Image with axes drawn on the chessboard
        """
        camera_matrix, dist_coeffs, _, _ = self.__load_calibration_data(params_path)
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
        # find corners for test image
        ret, corners = cv2.findChessboardCorners(gray, (self.height, self.width), None)
        
        if ret:
            # refine corner detection
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                        
            # find rotation and translation vectors
            ret, rvecs, tvecs = cv2.solvePnP(self.obj_points, corners, camera_matrix, dist_coeffs)
            
            # define axis points (origin and points along x, y, z axes)
            axis_length = 6 * self.square_size  # Length of axes in same units as square_size
            axis_points = np.float32([[0,0,0], 
                                    [axis_length,0,0], 
                                    [0,axis_length,0], 
                                    [0,0,-axis_length]])

            # define cube points
            cube_length = 3 * self.square_size
            cube_points = np.float32([[0,0,0], [0,cube_length,0], [cube_length,cube_length,0], [cube_length,0,0],
                            [0,0,-cube_length],[0,cube_length,-cube_length],[cube_length,cube_length,-cube_length],[cube_length,0,-cube_length] ])
            
            # project axis points to image plane
            axis_imgpts, _ = cv2.projectPoints(axis_points, rvecs, tvecs, camera_matrix, dist_coeffs)
            # project cube points to image plane
            cube_imgpts, _ = cv2.projectPoints(cube_points, rvecs, tvecs, camera_matrix, dist_coeffs)

            # define axis colors
            color_x = (90, 90, 219)
            color_y = (124, 219, 90)
            color_z = (219, 194, 90)

            # draw axis lines
            origin = tuple(map(int, axis_imgpts[0].ravel()))
            cv2.arrowedLine(img, origin, tuple(map(int, axis_imgpts[1].ravel())), color_x, 2, tipLength=0.03)
            cv2.arrowedLine(img, origin, tuple(map(int, axis_imgpts[2].ravel())), color_y, 2, tipLength=0.03)
            cv2.arrowedLine(img, origin, tuple(map(int, axis_imgpts[3].ravel())), color_z, 2, tipLength=0.03)
            
            # add axis labels
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            offset = 15
            img = cv2.putText(img, 'x', tuple(map(int, axis_imgpts[1].ravel() + offset)), font, 1, color_x, 2)
            img = cv2.putText(img, 'y', tuple(map(int, axis_imgpts[2].ravel() + offset)), font, 1, color_y, 2)
            img = cv2.putText(img, 'z', tuple(map(int, axis_imgpts[3].ravel() + offset)), font, 1, color_z, 2)

            # define cube colors
            top_shade = (153, 255, 204)
            pillar_shade = (0, 153, 255)

            cube_imgpts = np.int32(cube_imgpts).reshape(-1,2)
            
            # draw pillars
            for i,j in zip(range(4),range(4,8)):
                img = cv2.line(img, tuple(cube_imgpts[i]), tuple(cube_imgpts[j]), pillar_shade, 1)
            
            # draw bottom borders
            img = cv2.drawContours(img, [cube_imgpts[:4]], -1, pillar_shade, 1)

            # shade the top layer
            img = cv2.drawContours(img, [cube_imgpts[4:]], -1, pillar_shade, -1)

            # draw top borders
            img = cv2.drawContours(img, [cube_imgpts[4:]], -1, pillar_shade, 1)

            # display image
            cv2.imshow('Axes and Cube on Chessboard', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if save:
                fname = os.path.basename(params_path).split('.')[0]
                save_path = os.path.join(save_root, f'{fname}.png')
                cv2.imwrite(save_path, img)
                print(f"Image with axes drawn saved to {save_path}")
            
            return img
        else:
            raise Exception("Chessboard corners not found!")

    def plot_camera_locations(self, params_path):
        """
        Plots the 3D locations of the camera relative to the chessboard and the camera's viewing direction.

        Args:
            params_path: Path to the calibration parameters JSON file.
        """
        camera_matrix, dist_coeffs, rvecs, tvecs = self.__load_calibration_data(params_path)
        rvecs = [np.array(rvec) for rvec in rvecs]
        tvecs = [np.array(tvec) for tvec in tvecs]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # --- Plot the chessboard origin ---
        # Use the corner as the origin (0, 0, 0)
        ax.scatter(0, 0, 0, c='red', marker='o', label='Chessboard Origin')  # Plot origin
        # --- Plot the x,y,z axes ---
        axis_length = 30  # Length of axes in mm
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', label='X Axis')  # X-axis
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', label='Y Axis')  # Y-axis
        ax.quiver(0, 0, 0, 0, 0, -axis_length, color='b', label='Z Axis')  # Z-axis

        # --- Plot camera poses ---
        for i in range(len(rvecs)):
            R, _ = cv2.Rodrigues(rvecs[i])  # Convert rotation vector to rotation matrix
            # Camera position is the inverse transformation of the chessboard pose
            camera_position = -np.dot(R.T, tvecs[i].reshape(3, 1))
            ax.scatter(camera_position[0], camera_position[1], camera_position[2], marker='^', label=f'Camera {i+1}')
        
            # --- Plot the point the camera is looking at ---
            # Assume the camera is looking at the center of the chessboard
            board_width = (self.width - 1) * self.square_size
            board_height = (self.height - 1) * self.square_size
            center_chessboard = np.array([board_width / 2, board_height / 2, 0])

            # Transform the chessboard center back to the world coordinate
            ax.scatter(center_chessboard[0], center_chessboard[1], center_chessboard[2], marker='x', color='blue', label=f'Camera {i+1} Looking At')

        ax.set_title('Camera Locations Relative to Chessboard')
        ax.legend()

        # --- Plot the chessboard plane ---
        board_width = (self.width - 1) * self.square_size
        board_height = (self.height - 1) * self.square_size
        x = np.linspace(0, board_width, 10)
        y = np.linspace(0, board_height, 10)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        ax.plot_surface(x, y, z, alpha=0.2, color='gray', label='Chessboard Plane')

        plt.show()


    def calibrate(self, display=False, save=True):
        for image_path in self.image_paths:
            # read the image and convert it to grayscale
            img = cv2.imread(image_path)
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(
                gray_img, (self.height, self.width), None
            )
            
            # if the corners are found, refine them using cornerSubPix
            if ret == True:
                local_corners = cv2.cornerSubPix(
                    gray_img, corners, (11, 11), (-1, -1), self.criteria
                )
            # if the corners are not found, compute them manually
            else:
                local_corners = self.__compute_local_corners(img)

            # append the object points and image points
            self.global_obj_points.append(self.obj_points)
            self.global_img_points.append(local_corners)

            # display the result if needed
            if display:
                cv2.drawChessboardCorners(
                    img, (self.height, self.width), local_corners, True
                )
                cv2.imshow(os.path.basename(image_path), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()   
            

        # calibrate the camera with no flags for default behavior of not fixing cx, cy, fx, or fy
        # this is not needed but ensures CALIB_FIX_PRINCIPAL_POINT and CALIB_FIX_FOCAL_LENGTH are not set
        c_ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.global_obj_points,
            self.global_img_points,
            gray_img.shape[::-1],
            None,
            None,
            flags=0,
        )

        print("Calibration successful." if c_ret else "Calibration failed.")

        if c_ret and save:
            print("Calibration successful. Saving data...")
            save_root = "./output"
            save_path = os.path.join(save_root, os.path.basename(self.image_root))
            self.__save_calibration_data(
                camera_matrix, dist_coeffs, rvecs, tvecs, save_path
            )


if __name__ == "__main__":
    # OFFLINE STEP:
    root_path = "./images/"
    # for i in range(1, 4):
    #     IMAGE_PATH = os.path.join(root_path, f"run_{i}")
    #     calibrator = CameraCalibrator(IMAGE_PATH, (6, 9), TILE_SIZE)
    #     calibrator.calibrate(display=False, save=True)
    #     print(f"Calibration for Run {i} complete.")

    # ONLINE STEP:
    # Load the calibration data
    TEST_IMG_PATH = os.path.join(root_path, "test")
    calibrator = CameraCalibrator(TEST_IMG_PATH, (6, 9), TILE_SIZE)
    for i in range(1, 4):
        params_path = f"./output/run_{i}.json"
        # calibrator.draw_axes_on_chessboard(os.path.join(TEST_IMG_PATH, f"test.png"), params_path, save=True)
        calibrator.plot_camera_locations(params_path)

        
        
