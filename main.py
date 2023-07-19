import cv2
import numpy as np
import time

# ArUco의 사전을 생성합니다
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
## dectector 매개변수 생성
parameters = cv2.aruco.DetectorParameters_create()


# 카메라 보정 값들이 있으면 이를 사용해야 좋습니다. 지금은 임의의 값들을 사용합니다.
## 카메라 보정 매개변수 설정
## 카메라 내부 매개변수 3x3
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
## 카메라 왜곡 계수. 렌즈 왜곡을 보정하기 위해 사용, 임의의 값으로 초기화된 상태
dist_coeffs = np.zeros((4, 1))  # 실제 적용에서는 카메라 보정을 통해 얻어야합니다.

# 비디오 캡쳐를 시작합니다.
cap = cv2.VideoCapture(0)
## 비디오 프레임 크기(wide X height) 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# fps 계산을 위한 초기 시간값을 저장합니다.
t1 = time.time()

while (True):
    # 1
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 마커를 인식합니다.
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # corners, ids, rejectedImgPoints = cv2.aruco.ArucoDetector(gray, aruco_dict, parameters=parameters)

    # 인식된 마커가 있을 때, 각 마커에 대해 처리합니다.
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        for i in range(ids.size):
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

    # fps 계산합니다.
    t2 = time.time()
    fps = 1 / (t2 - t1)
    t1 = t2

    # 계산된 fps를 화면에 표시합니다.
    cv2.putText(frame, "FPS: %.2f" % fps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    # 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
