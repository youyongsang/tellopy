import time
import cv2
import numpy as np
from threading import Thread
from djitellopy import Tello

# 드론 연결 및 설정
tello = Tello()
tello.connect()

keepRecording = True
tello.streamon()
frame_read = tello.get_frame_read()

def videoRecorder():
    # 비디오 파일로 저장 설정 (XVID 코덱, 30FPS, 프레임 크기 설정)
    height, width, _ = frame_read.frame.shape
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        video.write(frame_read.frame)
        time.sleep(1 / 30)

    video.release()

# 영상 녹화 스레드 시작
recorder = Thread(target=videoRecorder)
recorder.start()

# 드론 이륙
tello.takeoff()

show_contours = False  # 윤곽선 표시 여부 플래그

while True:
    img = frame_read.frame  # 드론의 현재 프레임 가져오기
    
    if show_contours:
        # 실시간으로 윤곽선 검출
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 윤곽선 표시

    cv2.imshow("drone", img)  # 화면 출력

    key = cv2.waitKey(1) & 0xff
    if key == 27:  # ESC 키를 누르면 착륙 후 종료
        break
    elif key == ord('w'):
        tello.move_forward(30)
    elif key == ord('s'):
        tello.move_back(30)
    elif key == ord('a'):
        tello.move_left(30)
    elif key == ord('d'):
        tello.move_right(30)
    elif key == ord('e'):
        tello.rotate_clockwise(30)
    elif key == ord('q'):
        tello.rotate_counter_clockwise(30)
    elif key == ord('r'):
        tello.move_up(30)
    elif key == ord('f'):
        tello.move_down(30)
    elif key == 32:  # 스페이스바를 누르면 윤곽선 검출 토글
        show_contours = not show_contours

# 드론 착륙 및 종료
tello.land()
keepRecording = False
recorder.join()
cv2.destroyAllWindows()
