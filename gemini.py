import os
import cv2
import numpy as np
import glob

class CameraCalibrator:
    def __init__(self, images_root, chessboard_size, square_size):
        self.width, self.height = chessboard_size
        self.square_size = square_size
        self.image_root = images_root
        self.image_paths = glob.glob(images_root + '/*.jpg')
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.obj_points = np.zeros((self.width*self.height, 3), np.float32)
        self.obj_points[:, :2] = np.mgrid[0:self.height, 0:self.width].T.reshape(-1, 2) * self.square_size
        self.obj_points = [self.obj_points]  # Make it a list so we can multiply it later

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

    def calibrate(self):
        # Separate training and test images.  Assume the last image is the test image.
        training_images = self.image_paths[:-1]
        test_image_path = self.image_paths[-1]

        # Data lists for calibration
        global_obj_points = []
        global_img_points = []
        manual_count = 0

        for image_path in training_images:
            img = cv2.imread(image_path)
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray_img, (self.height, self.width), None)

            if ret:
                local_corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), self.criteria)
            else:
                local_corners = self.__compute_local_corners(img)
                manual_count +=1

            global_obj_points.append(self.obj_points)
            global_img_points.append(local_corners)
            print(f"Processed: {image_path}, Automatic Detection: {ret}, Manual Count: {manual_count}")

        # Calibrate camera
        self.run_calibration(global_obj_points, global_img_points, "All 24 Training Images")

        # Run 2: Use only 10 images where corner points were found automatically.
        auto_obj_points = []
        auto_img_points = []
        auto_image_count = 0
        for image_path in training_images:
            img = cv2.imread(image_path)
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray_img, (self.height, self.width), None)

            if ret and auto_image_count < 10:
                local_corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), self.criteria)
                auto_obj_points.append(self.obj_points)
                auto_img_points.append(local_corners)
                auto_image_count += 1

        self.run_calibration(auto_obj_points, auto_img_points, "10 Automatically Detected Images")

        # Run 3: Use only 5 images from Run 2
        self.run_calibration(auto_obj_points[:5], auto_img_points[:5], "5 Automatically Detected Images")

        # Process the test image (find corners automatically)
        test_img = cv2.imread(test_image_path)
        test_bgr_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        test_gray_img = cv2.cvtColor(test_bgr_img, cv2.COLOR_BGR2GRAY)
        ret, test_corners = cv2.findChessboardCorners(test_gray_img, (self.height, self.width), None)

        if ret:
            test_local_corners = cv2.cornerSubPix(test_gray_img, test_corners, (11, 11), (-1, -1), self.criteria)
            print("Test image corners found automatically.")
        else:
            print("Could not find chessboard corners in test image automatically.")
            # Consider handling the case where the test image also requires manual annotation.

        # Save intrinsics and extrinsics to text file
        camera_matrix, dist_coeffs, rvecs, tvecs = self.run_calibration(global_obj_points, global_img_points, "All 24 Training Images", save=True)

        print("Calibration complete.  Check the output.")

    def run_calibration(self, obj_points, img_points, run_name, save=False):
        """Calibrates the camera and prints/stores the results."""
        # Adjust the shape of the object points
        obj_points = [obj * len(img_points) for obj in self.obj_points]

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (640, 480), # Assuming a default image size.  Replace with your actual image size!
            None, None,
            flags=0 #cv2.CALIB_RATIONAL_MODEL
        )

        print(f"Calibration Run: {run_name}")
        print(f"  Reprojection Error: {ret}")
        print(f"  Camera Matrix:\n{camera_matrix}")
        print(f"  Distortion Coefficients:\n{dist_coeffs}")

        if save:
            with open('camera_params.txt', 'w') as f:
                f.write(f"Camera Matrix:\n{camera_matrix}\n")
                f.write(f"Distortion Coefficients:\n{dist_coeffs}\n")
                f.write(f"Rotation Vectors:\n{rvecs}\n")
                f.write(f"Translation Vectors:\n{tvecs}\n")
                print(f"Camera parameters saved to camera_params.txt")

        return camera_matrix, dist_coeffs, rvecs, tvecs


# Example Usage:
if __name__ == "__main__":
    # Replace with your actual values!
    image_dir = "images"  # Directory containing the chessboard images
    chessboard_size = (7, 9)  # (width, height) - number of INNER corners
    square_size = 25.0 # Size of each square in mm (measured!)

    calibrator = CameraCalibrator(image_dir, chessboard_size, square_size)
    calibrator.calibrate()