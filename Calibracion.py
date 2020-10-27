'''
/////////////////////////////////////////////
//    PONTIFICIA UNIVERSIDAD JAVERIANA     //
//                                         //
//  Carlos Daniel Cadena Cahvarro          //
//  Carlos Arturo Redondo Hurtado          //
//  Procesamiento de imagenes y vision     //
//  TALLER #2                              //
/////////////////////////////////////////////
'''

import numpy as np
import cv2
import glob
import os
import json

w = 7
h = 6

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((h * w, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

path = r'.\Camarapc' #r'.\Phone_images'
path_file = os.path.join(path, 'imagen*.jpeg')

images = glob.glob(path_file)
print(images)
for fname in images:
    img = cv2.imread(fname)
    #cv2.imshow('imgaas', img)
    #cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

    # If found, add object points, image points (after refining them)
    print(ret, corners)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (w, h), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(250)

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(mtx)
print(dist)

# reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))

file_name = 'calibration_pc.json'
json_file = os.path.join(path, file_name)

data = {
    'K': mtx.tolist(),
    'distortion': dist.tolist()
}

with open(json_file, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=1, ensure_ascii=False)

with open(json_file) as fp:
    json_data = json.load(fp)
print(json_data)
