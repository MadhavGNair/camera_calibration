import os
import cv2
import numpy as np
import glob
import json

TILE_SIZE = 16 # size of the chessboard square in mm

class CameraCalibrator:
    def __init__(self, images_root, chessboard_size, square_size):
        self.width, self.height = chessboard_size
        self.square_size = square_size
        self.image_root = images_root
        self.image_paths = glob.glob(images_root + '/*.png')
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.obj_points = np.zeros((self.width*self.height, 3), np.float32)
        self.obj_points[:, :2] = np.mgrid[0:self.height, 0:self.width].T.reshape(-1, 2) * self.square_size
        self.global_obj_points = []
        self.global_img_points = []

    def __get_mouse_click(self, event, x, y, flags, param):
        corners = param['global_corners']
        win_title = param['win_title']
        image = param['image']
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(win_title, image)
            if len(corners) >= 4:
                cv2.destroyWindow(win_title)
        return corners

    def __select_global_corners(self, image):
        win_title = "Select Corners in order top-left, top-right, bottom-right, bottom-left"
        global_corners = []
        img = image.copy()

        params = {'image': img, 'global_corners': global_corners, 'win_title': win_title}      
        
        cv2.namedWindow(win_title)
        cv2.setMouseCallback(win_title, self.__get_mouse_click, params)
        cv2.imshow(win_title, image)
        cv2.waitKey(0)
        return global_corners

    def __compute_local_corners(self, image):
        # select the global corners of the chessboard manually
        global_corners = self.__select_global_corners(image)
        tl, tr, br, bl = global_corners

        # compute the top and left edges of the chessboard
        top_edges = np.array(tr) - np.array(tl)
        left_edges = np.array(bl) - np.array(tl)
        tl = np.array(tl)

        # get the local corners of the chessboard
        local_corners = np.zeros((self.width * self.height, 1, 2), np.float32)
        for i in range(self.height):
            for j in range(self.width):
                # compute the point using the top and left edges
                point = tl + j * self.square_size * (top_edges / (self.square_size * (self.width - 1))) + i * self.square_size * (left_edges / (self.square_size * (self.height - 1)))
                local_corners[i * self.width + j, 0] = point
        return local_corners

    def __save_calibration_data(self, cam_matrix, coeffs, rvecs, tvecs, filepath):
        if cam_matrix is None or coeffs is None or rvecs is None or tvecs is None:
            raise ValueError("Calibration data not fully available. Run calibrate() first.")

        calibration_data_json = {
            "camera_matrix": cam_matrix.tolist(),
            "dist_coeffs": coeffs.tolist(),
            "rvecs": [rvec.tolist() for rvec in rvecs],
            "tvecs": [tvec.tolist() for tvec in tvecs]
        }
        with open(filepath + ".json", "w") as f:
            json.dump(calibration_data_json, f, indent=4)
        print(f"Calibration data saved to {filepath}.json")

    def calibrate(self, display=False):
        for image_path in self.image_paths:
            # skip the hard images for now
            if os.path.basename(image_path)[:1] == 'h':
                continue

            img = cv2.imread(image_path)
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray_img, (self.height, self.width), None)
            # if the corners are found, refine them using cornerSubPix
            if ret == True:
                local_corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), self.criteria)
            # if the corners are not found, compute them manually
            else:
                local_corners = self.__compute_local_corners(img)

            # append the object points and image points
            self.global_obj_points.append(self.obj_points)
            self.global_img_points.append(local_corners)

            # display the result if needed
            if display:
                cv2.drawChessboardCorners(img, (self.height, self.width), local_corners, True)
                cv2.imshow(os.path.basename(image_path), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
        # calibrate the camera with no flags for default behavior of not fixing cx, cy, fx, or fy 
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(self.global_obj_points, self.global_img_points, gray_img.shape[::-1], None, None, flags=0)

        if ret:
            print("Calibration successful. Saving data...")
            save_root = "./output"
            save_path = os.path.join(save_root, os.path.basename(self.image_root))
            self.__save_calibration_data(camera_matrix, dist_coeffs, rvecs, tvecs, save_path)
        

if __name__ == '__main__':
    root_path = './images/'
    for i in range(1, 4):
        IMAGE_PATH = os.path.join(root_path, f'run_{i}')
        calibrator = CameraCalibrator(IMAGE_PATH, (6, 9), TILE_SIZE)
        calibrator.calibrate()
        print(f"Calibration for Run {i} complete.")
