import numpy as np
import cv2
import glob

def prepare_object_points(square_size, pattern_size):
    object_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    object_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    object_points *= square_size
    return object_points

def process_images(images_path, pattern_size, object_points):
    images_list = glob.glob(images_path)
    points_on_real_world = []
    points_on_camera_plan = []
    
    for fname in images_list:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            points_on_real_world.append(object_points)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            points_on_camera_plan.append(corners2)
            img = cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('Img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    return points_on_real_world, points_on_camera_plan, gray.shape[::-1]

def calibrate_camera(points_on_real_world, points_on_camera_plan, image_shape):
    return cv2.calibrateCamera(points_on_real_world, points_on_camera_plan, image_shape, None, None)

def save_kitti_format(camera_matrix, filename):
    P0 = np.zeros((3, 4))
    P0[:, :3] = camera_matrix
    P0_str = ' '.join(f'{num:.12e}' for num in P0.flatten())
    with open(filename, 'w') as f:
        f.write(f'P0: {P0_str}\n')
    print(f'P0: {P0_str}\n')

def calculate_reprojection_error(points_on_real_world, points_on_camera_plan, rotation_vectors, translation_vectors, camera_matrix, dist_coeffs):
    mean_error = 0
    for i in range(len(points_on_real_world)):
        image_points2, _ = cv2.projectPoints(points_on_real_world[i], rotation_vectors[i], translation_vectors[i], camera_matrix, dist_coeffs)
        error = cv2.norm(points_on_camera_plan[i], image_points2, cv2.NORM_L2) / len(image_points2)
        mean_error += error
    mean_error /= len(points_on_real_world)
    return mean_error

def undistort_image(image_path, saved_params):
    with np.load(saved_params) as X:
        camera_matrix, dist_coeffs = X['mtx'], X['dist']
    img = cv2.imread(image_path)
    original_img = img.copy()
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imshow('Original Image', original_img)
    cv2.imshow('Calibrated Image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    square_size = 2.5 # cm
    pattern_size = (9, 6)
    images_path = 'FotosPICameraCalibration/*.png'
    object_points = prepare_object_points(square_size, pattern_size)
    points_on_real_world, points_on_camera_plan, image_shape = process_images(images_path, pattern_size, object_points)
    calibration_error, camera_matrix, dist_coeffs, rotation_vectors, translation_vectors = calibrate_camera(points_on_real_world, points_on_camera_plan, image_shape)
    save_kitti_format(camera_matrix, 'CalibrationCam/ParameteresCamera.txt')
    mean_error = calculate_reprojection_error(points_on_real_world, points_on_camera_plan, rotation_vectors, translation_vectors, camera_matrix, dist_coeffs)
    print("Calibration erro:", calibration_error)
    print("Erro médio de reprojeção:", mean_error)
    np.savez('calibrationDataPICamera', mtx=camera_matrix, dist=dist_coeffs, rvecs=rotation_vectors, tvecs=translation_vectors)
    undistort_image('FotosPICameraCalibration/image_0014.png', 'calibrationDataPICamera.npz')

if __name__ == '__main__':
    main()
