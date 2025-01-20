# -*- coding:utf-8 -*-
import cv2
import numpy as np
from djitellopy import Tello

# 드론 연결 및 설정
tello = Tello()
tello.connect()

tello.streamon()  # 드론 카메라 스트리밍 시작
frame_read = tello.get_frame_read()

tello.takeoff()  # 드론 이륙

while True:
    img = frame_read.frame  # 현재 드론 카메라 프레임 가져오기
    cv2.imshow("drone", img)

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
    
    # 스페이스바를 누르면 윤곽선 검출 코드 실행
    elif key == 32:  # Spacebar key
        cv2.imwrite("captured_image.jpg", img)  # 현재 프레임 저장

        # 이미지를 불러와 윤곽선 검출 수행
        img = cv2.imread("captured_image.jpg")
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 임계값을 적용하여 이진화 처리
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)

        # 윤곽선 검출
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 검출된 윤곽선을 이미지에 그림
        image_with_contours = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

        # 윤곽선 검출된 이미지 출력
        cv2.imshow("Contours", image_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

tello.land()  # 드론 착륙
cv2.destroyAllWindows()  # 모든 창 닫기
