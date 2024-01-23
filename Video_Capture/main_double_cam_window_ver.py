import cv2
import numpy as np
import multiprocessing as mp
import time

# ArUco의 사전을 생성합니다
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# 카메라 보정 값들이 있으면 이를 사용해야 좋습니다. 지금은 임의의 값들을 사용합니다.
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # 카메라 왜곡 계수. 실제 적용에서는 카메라 보정을 통해 얻어야합니다.


def video_capture_process(index):
    cap = cv2.VideoCapture(index)

    # fps 계산을 위한 초기 시간값을 저장합니다.
    t1 = time.time()

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 마커를 인식합니다.
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

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
        cv2.putText(frame, f"Camera {index} FPS: %.2f" % fps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(f'Camera {index}', frame)

        # 'q'키를 누르면 비디오 캡처가 종료됩니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 프로세스를 생성합니다.
    p1 = mp.Process(target=video_capture_process, args=(0,))
    p2 = mp.Process(target=video_capture_process, args=(1,))

    # 프로세스를 시작합니다.
    p1.start()
    p2.start()

    # 프로세스가 끝날 때까지 기다립니다.
    p1.join()
    p2.join()
