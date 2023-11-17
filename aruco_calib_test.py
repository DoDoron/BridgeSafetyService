import numpy as np
import cv2
from cv2 import aruco
import os
import pandas as pd
import glob

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def readChessboards(images):
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        img=cv2.imread(im)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, aruco_dict)

        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize=(3, 3),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator += 1

    imsize = gray.shape
    return allCorners, allIds, imsize

def calibrateCamera(allCorners, allIds, imsize, board):
    cameraMatrixInit = np.array([[1000.,    0., imsize[0]/2.],
                                 [0., 1000., imsize[1]/2.],
                                 [0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5, 1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
             cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    # flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors, perViewErrors


if __name__ == "__main__":
    # ArUco 딕셔너리 생성
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
    # 랜덤 마커 보드 생성
    board = aruco.CharucoBoard_create(7, 6, 1, .8, aruco_dict)
    # 보드 이미지 생성
    # imboard = board.draw((2000, 2000))
    # 주어진 디렉터리에 저장된 개별 이미지의 경로 추출
    images = glob.glob('./Calibration_dataset/*.jpg')

    allCorners, allIds, imsize = readChessboards(images)
    ret, mtx, dst, rvecs, tvecs, perViewErrors = calibrateCamera(
        allCorners, allIds, imsize, board)
    
    createDirectory("const_test")
    df_mtx = pd.DataFrame(mtx)
    df_mtx.to_csv("const/mtx.csv", index=False, header=False)
    df_dst = pd.DataFrame(dst)
    df_dst.to_csv("const/dst.csv", index=False, header=False)
    df_err = pd.DataFrame(perViewErrors)
    df_err.to_csv("const/err.csv", index=False, header=False)