from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import threading
from queue import Queue
import os
from djitellopy import Tello
import time
from datetime import datetime
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from gtts import gTTS
import pygame
import tempfile
from PIL import Image
from ultralytics import YOLO
from smolagents import ToolCallingAgent, LiteLLMModel, tool
from typing import Optional

# .env 파일 로드
load_dotenv()

# Google Gemini 설정
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro-vision')

# AI 에이전트 설정
llm_model = LiteLLMModel(model_id="gemini/gemini-2.0-flash-exp",
                        api_key=os.getenv("GOOGLE_API_KEY"))

# YOLO 모델 로드
yolo_model = YOLO('yolov8n.pt')

app = Flask(__name__)

class TelloController:
    def __init__(self):
        self.tello = Tello()
        self.frame_reader = None
        self.is_streaming = False
        self.frame_queue = Queue(maxsize=10)
        self.stream_thread = None
        self.is_flying = False
        self.tracking_mode = False
        self.person_detected = False
        pygame.mixer.init()
        self.setup_drone_control_tools()

    def analyze_person_state(self, frame):
        """감지된 사람의 상태를 분석"""
        try:
            # 이미지를 임시 파일로 저장
            temp_path = f'photos/temp_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            cv2.imwrite(temp_path, frame)
            
            # Gemini Vision으로 이미지 분석
            try:
                image = Image.open(temp_path)
                response = model.generate_content([
                    "이 이미지에서 보이는 사람의 상태를 분석해주세요. 특히 누워있거나, 쓰러져있거나, 다친 것처럼 보이는 사람이 있는지 자세히 설명해주세요.",
                    image
                ])
                analysis = response.text
            except Exception as e:
                print(f"Gemini 분석 오류: {str(e)}")
                return None
            
            # 임시 파일 삭제
            os.remove(temp_path)
            
            return analysis
        except Exception as e:
            print(f"상태 분석 오류: {str(e)}")
            return None

    def search_for_person(self, condition: str) -> str:
        """특정 상태의 사람을 찾아서 분석"""
        if not self.is_flying:
            return "드론이 이륙하지 않았습니다. 먼저 이륙해주세요."
        
        try:
            # 360도 회전하면서 사람 탐지
            for _ in range(4):  # 90도씩 4번 회전
                frame = self.frame_reader.frame
                if frame is not None:
                    results = yolo_model(frame)
                    
                    for result in results:
                        for box in result.boxes:
                            if box.cls == 0:  # person class
                                # 사람이 감지되면 상태 분석
                                analysis = self.analyze_person_state(frame)
                                if analysis:
                                    # TTS로 결과 읽기
                                    try:
                                        self.speak(analysis)
                                    except Exception as e:
                                        print(f"TTS 오류: {str(e)}")
                                    return f"사람 발견! 분석 결과: {analysis}"
                
                # 90도 회전
                self.rotate("clockwise", 90)
                time.sleep(2)  # 회전 후 안정화 대기
            
            return "주변에서 사람을 찾지 못했습니다."
            
        except Exception as e:
            return f"탐색 중 오류 발생: {str(e)}"

    def setup_drone_control_tools(self):
        """드론 제어를 위한 도구들을 설정"""
        
        @tool
        def move_drone(direction: str, distance: Optional[int] = 30) -> str:
            """
            드론을 지정된 방향으로 이동시킵니다.
            Args:
                direction: 이동 방향 ('up': 위로, 'down': 아래로, 'left': 왼쪽으로, 
                          'right': 오른쪽으로, 'forward': 앞으로, 'back': 뒤로)
                distance: 이동 거리 (cm), 기본값 30cm
            Returns:
                str: 이동 결과 메시지
            """
            try:
                self.move(direction, distance)
                return f"드론을 {direction} 방향으로 {distance}cm 이동했습니다."
            except Exception as e:
                return f"드론 이동 중 오류 발생: {str(e)}"

        @tool
        def rotate_drone(direction: str, angle: Optional[int] = 90) -> str:
            """
            드론을 지정된 방향으로 회전시킵니다.
            Args:
                direction: 회전 방향 ('clockwise': 시계 방향, 'counter_clockwise': 반시계 방향)
                angle: 회전 각도 (도 단위), 기본값 90도
            Returns:
                str: 회전 결과 메시지
            """
            try:
                self.rotate(direction, angle)
                return f"드론을 {direction} 방향으로 {angle}도 회전했습니다."
            except Exception as e:
                return f"드론 회전 중 오류 발생: {str(e)}"

        @tool
        def takeoff_drone() -> str:
            """드론을 이륙시킵니다."""
            try:
                if not self.is_flying:
                    self.takeoff()
                    return "드론이 이륙했습니다."
                return "드론이 이미 비행 중입니다."
            except Exception as e:
                return f"이륙 중 오류 발생: {str(e)}"

        @tool
        def land_drone() -> str:
            """드론을 착륙시킵니다."""
            try:
                if self.is_flying:
                    self.land()
                    return "드론이 착륙했습니다."
                return "드론이 이미 착륙한 상태입니다."
            except Exception as e:
                return f"착륙 중 오류 발생: {str(e)}"

        @tool
        def search_person(condition: str) -> str:
            """
            특정 상태의 사람을 찾아서 분석합니다.
            Args:
                condition: 찾고자 하는 사람의 상태 설명 (예: '누워있는', '다친', '쓰러진' 등)
            """
            try:
                return self.search_for_person(condition)
            except Exception as e:
                return f"사람 탐색 중 오류 발생: {str(e)}"

        # AI 에이전트 초기화
        self.agent = ToolCallingAgent(
            tools=[
                move_drone, 
                rotate_drone, 
                takeoff_drone, 
                land_drone,
                search_person
            ],
            model=llm_model,
            system_prompt="""{{managed_agents_descriptions}}
            당신은 Tello 드론을 제어하는 AI 어시스턴트입니다. 
            사용자의 자연어 명령을 해석하여 적절한 드론 제어 명령으로 변환하세요.
            
            가능한 기능:
            - 이륙/착륙
            - 이동 (상/하/좌/우/전진/후진)
            - 회전 (시계/반시계 방향)
            - 특정 상태의 사람 찾기 (누워있는 사람, 다친 사람 등)
            
            사람 찾기 명령을 받으면, 드론은 자동으로 주변을 탐색하며 
            YOLO로 사람을 감지하고 Gemini Vision으로 상태를 분석합니다.
            """
        )

    def process_ai_command(self, command: str) -> str:
        """AI 에이전트를 통해 자연어 명령을 처리"""
        try:
            response = self.agent.run(command)
            return response
        except Exception as e:
            return f"명령 처리 중 오류가 발생했습니다: {str(e)}"

    def connect(self):
        """드론 연결 및 상태 확인"""
        try:
            # 기존 연결이 있다면 정리
            if self.is_streaming:
                self.stop_video_stream()
            
            print("드론에 연결 중...")
            
            # 연결 전 UDP 포트 초기화 및 타임아웃 설정
            self.tello.RESPONSE_TIMEOUT = 15  # 타임아웃 증가
            self.tello.RETRY_COUNT = 5       # 재시도 횟수 증가
            
            # 드론 연결 시도
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    print(f"연결 시도 {retry_count + 1}/{max_retries}...")
                    self.tello.connect()
                    time.sleep(3)  # 연결 안정화 대기 시간 증가
                    
                    # 연결 확인을 위한 명령 전송
                    self.tello.send_command_with_return("command")
                    print("✓ 명령 모드 활성화 성공!")
                    
                    # 드론 상태 확인
                    battery = self.tello.get_battery()
                    print(f"✓ 배터리 잔량: {battery}%")
                    
                    if battery < 20:
                        raise Exception("배터리가 너무 부족합니다 (20% 미만)")
                    
                    print("✓ 연결 성공!")
                    return True
                    
                except Exception as e:
                    print(f"연결 시도 {retry_count + 1} 실패: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print("3초 후 재시도...")
                        time.sleep(3)
                    else:
                        raise Exception(f"드론 연결 실패: {str(e)}")
            
        except Exception as e:
            print(f"연결 오류: {str(e)}")
            raise

    def stop_video_stream(self):
        """비디오 스트리밍 중지"""
        print("비디오 스트림 정지 중...")
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        try:
            self.tello.streamoff()
        except:
            pass
        # 큐 비우기
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass

    def detect_and_track_person(self, frame):
        """사람 감지 및 추적"""
        try:
            # YOLO로 객체 감지
            results = yolo_model(frame)
            
            # 사람 클래스 (0번)를 찾음
            person_detected = False
            center_x = frame.shape[1] // 2  # 프레임 중앙 X 좌표
            
            for result in results:
                for box in result.boxes:
                    if box.cls == 0:  # person class
                        person_detected = True
                        # 박스의 중심점 계산
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        box_center_x = (x1 + x2) / 2
                        
                        if self.tracking_mode:
                            # 추적 로직
                            margin = 50  # 허용 오차 범위 (픽셀)
                            
                            if box_center_x < center_x - margin:
                                # 드론을 왼쪽으로 회전
                                self.tello.rotate_counter_clockwise(10)
                            elif box_center_x > center_x + margin:
                                # 드론을 오른쪽으로 회전
                                self.tello.rotate_clockwise(10)
                            
                            # 박스 크기로 거리 추정
                            box_width = x2 - x1
                            if box_width < 150:  # 대상이 너무 멀리 있음
                                self.tello.move_forward(30)
                            elif box_width > 250:  # 대상이 너무 가까이 있음
                                self.tello.move_back(30)
                        break
            
            self.person_detected = person_detected
            
        except Exception as e:
            print(f"추적 오류: {str(e)}")

    def toggle_tracking_mode(self):
        """추적 모드 전환"""
        self.tracking_mode = not self.tracking_mode
        return self.tracking_mode

    def start_video_stream(self):
        """비디오 스트리밍 시작"""
        if not self.is_streaming:
            try:
                print("비디오 스트림 시작 중...")
                self.tello.streamon()
                time.sleep(3)  # 스트림 초기화 대기 시간 증가
                
                retry_count = 0
                while retry_count < 3:
                    self.frame_reader = self.tello.get_frame_read()
                    if self.frame_reader and self.frame_reader.frame is not None:
                        break
                    print("프레임 리더 초기화 재시도...")
                    time.sleep(1)
                    retry_count += 1
                
                if not self.frame_reader or self.frame_reader.frame is None:
                    raise Exception("비디오 스트림 초기화 실패")
                
                self.is_streaming = True
                self.stream_thread = threading.Thread(target=self._stream_loop)
                self.stream_thread.daemon = True
                self.stream_thread.start()
                print("✓ 비디오 스트리밍 시작됨")
                
            except Exception as e:
                print(f"스트리밍 시작 오류: {str(e)}")
                self.is_streaming = False
                raise

    def _stream_loop(self):
        """비디오 스트리밍 루프"""
        while self.is_streaming:
            if self.frame_reader:
                frame = self.frame_reader.frame
                if frame is not None:
                    frame = cv2.resize(frame, (640, 480))
                    
                    # 사람 감지 및 추적
                    if self.is_flying:
                        self.detect_and_track_person(frame)
                    
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                    except:
                        pass
            time.sleep(0.03)

    def take_photo(self):
        """사진 촬영"""
        if not os.path.exists('photos'):
            os.makedirs('photos')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'photos/tello_scan_{timestamp}.jpg'
        
        frame = self.frame_reader.frame
        cv2.imwrite(filename, frame)
        print(f"사진 저장됨: {filename}")
        return filename, frame

    def create_panorama(self):
        """파노라마 촬영"""
        try:
            print("파노라마 촬영 시작...")
            images = []
            
            # 360도 회전하면서 사진 촬영 (90도씩 4장)
            for i in range(4):
                print(f"사진 {i+1}/4 촬영 중...")
                frame = self.frame_reader.frame
                if frame is not None:
                    images.append(frame.copy())
                else:
                    raise Exception("프레임을 가져올 수 없습니다")
                
                if i < 3:  # 마지막 사진 후에는 회전하지 않음
                    print(f"{90}도 회전 중...")
                    self.tello.rotate_clockwise(90)
                    time.sleep(2)  # 회전 후 안정화 대기
            
            print("파노라마 이미지 생성 중...")
            stitcher = cv2.Stitcher.create()
            status, panorama = stitcher.stitch(images)
            
            if status == cv2.Stitcher_OK:
                if not os.path.exists('panoramas'):
                    os.makedirs('panoramas')
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'panoramas/tello_panorama_{timestamp}.jpg'
                cv2.imwrite(filename, panorama)
                print(f"파노라마 저장됨: {filename}")
                return filename, "파노라마 이미지가 성공적으로 생성되었습니다."
            else:
                error_msg = f"파노라마 스티칭 실패 (status: {status})"
                print(error_msg)
                return None, error_msg
                
        except Exception as e:
            error_msg = f"파노라마 촬영 오류: {str(e)}"
            print(error_msg)
            return None, error_msg

    def analyze_image(self, image_path: str) -> str:
        """Gemini Vision으로 이미지 분석"""
        try:
            # PIL을 사용하여 이미지 로드 및 처리
            image = Image.open(image_path)
            
            # Gemini에 이미지 전송 및 분석 요청
            response = model.generate_content([
                "이 이미지에서 보이는 것을 자세히 설명해주세요.",
                image
            ])
            
            return response.text
        except Exception as e:
            print(f"이미지 분석 오류: {str(e)}")
            return f"이미지 분석 중 오류가 발생했습니다: {str(e)}"

    def speak(self, text: str):
        """텍스트를 음성으로 변환하여 재생"""
        try:
            tts = gTTS(text=text, lang='ko')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_filename = fp.name
                tts.save(temp_filename)
            
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            os.unlink(temp_filename)
        except Exception as e:
            print(f"TTS 오류: {str(e)}")

    def scan_surroundings(self):
        """현재 보이는 장면을 촬영하고 분석"""
        try:
            print("사진 촬영 중...")
            filename, _ = self.take_photo()
            
            print("이미지 분석 중...")
            # Gemini Vision으로 이미지 분석
            try:
                image = Image.open(filename)
                response = model.generate_content([
                    "이 이미지에서 보이는 것을 자세히 설명해주세요. 한국어로 답변해주세요. 말하듯이 줄글로 답변해.",
                    image
                ])
                analysis = response.text
            except Exception as e:
                print(f"Gemini 분석 오류: {str(e)}")
                return None, f"이미지 분석 중 오류가 발생했습니다: {str(e)}"
            
            print(f"분석 결과: {analysis}")
            
            try:
                self.speak(analysis)
            except Exception as e:
                print(f"TTS 오류: {str(e)}")
            
            return filename, analysis
        except Exception as e:
            print(f"스캔 오류: {str(e)}")
            return None, f"스캔 중 오류가 발생했습니다: {str(e)}"

    def takeoff(self):
        """드론 이륙"""
        try:
            print("이륙 준비...")
            if not self.is_streaming:
                raise Exception("드론이 연결되지 않았습니다.")
            
            print("이륙!")
            self.tello.takeoff()
            time.sleep(3)  # 이륙 완료 대기
            self.is_flying = True
            print("이륙 완료!")
        except Exception as e:
            print(f"이륙 오류: {str(e)}")
            raise

    def land(self):
        """드론 착륙"""
        print("착륙!")
        self.tello.land()
        self.is_flying = False

    def move(self, direction: str, distance: int):
        """드론 이동"""
        if not self.is_flying:
            raise Exception("드론이 이륙하지 않았습니다. 먼저 이륙해주세요.")
            
        # 전진 명령일 때 전방 거리 확인
        if direction == "forward":
            # 전방 거리 센서 값 읽기 (cm 단위)
            front_distance = self.tello.get_distance_tof()
            
            # 안전 거리 설정 (70cm)
            SAFE_DISTANCE = 70
            
            if front_distance < SAFE_DISTANCE:
                # 후진 거리 설정 (30cm)
                BACKUP_DISTANCE = 30
                print(f"경고! 사물이 너무 가까이에 있습니다! (거리: {front_distance}cm)")
                print(f"안전을 위해 {BACKUP_DISTANCE}cm 후진합니다.")
                self.tello.move_back(BACKUP_DISTANCE)
                raise Exception(f"경고! 사물이 너무 가까이에 있습니다! (거리: {front_distance}cm)")
            
        print(f"{direction} 방향으로 {distance}cm 이동")
        if direction == "up":
            self.tello.move_up(distance)
        elif direction == "down":
            self.tello.move_down(distance)
        elif direction == "left":
            self.tello.move_left(distance)
        elif direction == "right":
            self.tello.move_right(distance)
        elif direction == "forward":
            self.tello.move_forward(distance)
        elif direction == "back":
            self.tello.move_back(distance)

    def rotate(self, direction: str, angle: int):
        """드론 회전
        direction: clockwise, counter_clockwise
        angle: 회전 각도
        """
        print(f"{direction} 방향으로 {angle}도 회전")
        if direction == "clockwise":
            self.tello.rotate_clockwise(angle)
        else:
            self.tello.rotate_counter_clockwise(angle)

    def process_contours(self, frame):
        """이미지에서 윤곽선 검출"""
        try:
            # 그레이스케일 변환
            imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 이진화
            ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            
            # 윤곽선 검출
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # 원본 이미지에 윤곽선 그리기
            result = frame.copy()
            cv2.drawContours(result, contours, -1, (0, 255, 0), 3)
            
            return result
        except Exception as e:
            print(f"이미지 처리 오류: {str(e)}")
            return frame

    def capture_and_process(self):
        """사진 촬영 및 윤곽선 검출"""
        try:
            if not os.path.exists('processed'):
                os.makedirs('processed')
            
            # 원본 이미지 촬영
            frame = self.frame_reader.frame
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 윤곽선 검출
            processed_frame = self.process_contours(frame)
            
            # 처리된 이미지 저장
            filename = f'processed/contours_{timestamp}.jpg'
            cv2.imwrite(filename, processed_frame)
            
            return filename
        except Exception as e:
            print(f"이미지 처리 오류: {str(e)}")
            raise

# 전역 컨트롤러 인스턴스
controller = None

def get_frame():
    """프레임 스트리밍을 위한 제너레이터 함수"""
    while True:
        if controller and not controller.frame_queue.empty():
            frame = controller.frame_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/connect', methods=['POST'])
def connect_drone():
    global controller
    try:
        if controller is None:
            controller = TelloController()
        
        controller.connect()
        controller.start_video_stream()
        return jsonify({"status": "success", "message": "드론이 연결되었습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/analyze_uploaded', methods=['POST'])
def analyze_uploaded():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "파일이 없습니다."})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "선택된 파일이 없습니다."})
        
        if file:
            # 임시 파일로 저장
            temp_path = os.path.join('photos', 'temp_' + file.filename)
            file.save(temp_path)
            
            # Gemini로 분석
            try:
                image = Image.open(temp_path)
                response = model.generate_content([
                    "이 이미지에서 보이는 것을 자세히 설명해주세요.",
                    image
                ])
                analysis = response.text
            except Exception as e:
                analysis = f"이미지 분석 중 오류가 발생했습니다: {str(e)}"
            
            # 임시 파일 삭제
            os.remove(temp_path)
            
            return jsonify({
                "status": "success",
                "message": "이미지 분석이 완료되었습니다.",
                "analysis": analysis
            })
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/chat', methods=['POST'])
def chat_with_gpt():
    """GPT-4와 채팅"""
    try:
        message = request.json.get('message')
        if not message:
            return jsonify({"status": "error", "message": "메시지가 없습니다."})
        
        response = model.generate_content([
            "당신은 Tello 드론 제어 시스템의 AI 어시스턴트입니다. 드론 조종과 관련된 질문에 친절하게 답변해주세요.",
            message
        ])
        
        reply = response.text
        return jsonify({
            "status": "success",
            "message": "응답이 생성되었습니다.",
            "reply": reply
        })
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/ai_command', methods=['POST'])
def ai_command():
    """AI 에이전트를 통한 드론 제어 처리"""
    try:
        if not controller:
            return jsonify({"status": "error", "message": "드론이 연결되지 않았습니다."})
        
        command = request.json.get('command')
        if not command:
            return jsonify({"status": "error", "message": "명령이 없습니다."})
        
        response = controller.process_ai_command(command)
        return jsonify({
            "status": "success",
            "message": "명령이 처리되었습니다.",
            "response": response
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/scan', methods=['POST'])
def scan_surroundings():
    try:
        if controller:
            filename, analysis = controller.scan_surroundings()
            image_url = f'/photos/{os.path.basename(filename)}'
            return jsonify({
                "status": "success",
                "message": "스캔이 완료되었습니다.",
                "analysis": analysis,
                "image_url": image_url
            })
        return jsonify({"status": "error", "message": "드론이 연결되지 않았습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/panorama', methods=['POST'])
def create_panorama():
    try:
        if controller:
            filename, message = controller.create_panorama()
            if filename:
                return jsonify({
                    "status": "success",
                    "message": "파노라마 촬영이 완료되었습니다.",
                    "image_url": f'/panoramas/{os.path.basename(filename)}',
                    "analysis": message
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": message
                })
        return jsonify({"status": "error", "message": "드론이 연결되지 않았습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/panoramas/<path:filename>')
def serve_panorama(filename):
    return send_from_directory('panoramas', filename)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if controller:
            filename = controller.capture_and_process()
            return jsonify({
                "status": "success",
                "message": "이미지 처리가 완료되었습니다.",
                "image_url": f'/processed/{os.path.basename(filename)}'
            })
        return jsonify({"status": "error", "message": "드론이 연결되지 않았습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/toggle_tracking', methods=['POST'])
def toggle_tracking():
    """추적 모드 전환 처리"""
    try:
        if controller:
            tracking_enabled = controller.toggle_tracking_mode()
            return jsonify({
                "status": "success",
                "message": f"추적 모드가 {'활성화' if tracking_enabled else '비활성화'} 되었습니다.",
                "tracking_enabled": tracking_enabled
            })
        return jsonify({"status": "error", "message": "드론이 연결되지 않았습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/photos/<path:filename>')
def serve_photo(filename):
    return send_from_directory('photos', filename)

@app.route('/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory('processed', filename)

def ensure_template_exists():
    """템플릿 디렉토리와 파일이 존재하는지 확인하고 생성"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    template_path = os.path.join(template_dir, 'index2.html')
    if not os.path.exists(template_path):
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Tello Drone Scanner with Google Gemini</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        .header img {
            width: 32px;
            height: 32px;
        }
        .video-container {
            margin: 20px 0;
            text-align: center;
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin: 20px 0;
        }
        .upload-section {
            margin: 20px 0;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .upload-form {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-top: 10px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4285f4;
            color: white;
            border: none;
            border-radius: 5px;
        }
        button:hover {
            background-color: #3367d6;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        #status {
            margin: 20px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .success {
            background-color: #dff0d8;
            color: #3c763d;
        }
        .error {
            background-color: #f2dede;
            color: #a94442;
        }
        #analysis {
            margin: 20px 0;
            padding: 15px;
            background-color: #fff;
            border-radius: 5px;
            border: 1px solid #ddd;
            white-space: pre-wrap;
        }
        .preview-image {
            max-width: 100%;
            max-height: 300px;
            margin-top: 10px;
            display: none;
        }
        .drone-controls {
            margin: 20px 0;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .control-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        .person-alert {
            margin: 10px 0;
            background-color: rgba(255, 0, 0, 0.8);
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-align: center;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg" alt="Gemini Logo">
            <h1>Tello Drone Scanner with Google Gemini</h1>
        </div>
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </div>
        <div id="personAlert" class="person-alert">사람이 감지되었습니다!</div>
        <div class="controls">
            <button onclick="connectDrone()">드론 연결</button>
            <button onclick="scanSurroundings()">주변 스캔 (Gemini Vision)</button>
            <button onclick="createPanorama()">파노라마 촬영</button>
            <button onclick="processImage()">윤곽선 검출</button>
        </div>
        <div class="drone-controls">
            <h2>드론 제어</h2>
            <div class="control-grid">
                <button onclick="controlDrone('takeoff')">이륙</button>
                <button onclick="controlDrone('land')">착륙</button>
                <button onclick="controlDrone('move', {direction: 'up', distance: 50})">위로 50cm</button>
                <button onclick="controlDrone('move', {direction: 'down', distance: 50})">아래로 50cm</button>
                <button onclick="controlDrone('move', {direction: 'forward', distance: 50})">전진 50cm</button>
                <button onclick="controlDrone('move', {direction: 'back', distance: 50})">후진 50cm</button>
                <button onclick="controlDrone('move', {direction: 'left', distance: 50})">왼쪽 50cm</button>
                <button onclick="controlDrone('move', {direction: 'right', distance: 50})">오른쪽 50cm</button>
                <button onclick="controlDrone('rotate', {direction: 'clockwise', angle: 90})">시계방향 90도</button>
                <button onclick="controlDrone('rotate', {direction: 'counter_clockwise', angle: 90})">반시계방향 90도</button>
            </div>
        </div>
        <div class="upload-section">
            <h2>이미지 업로드 및 분석</h2>
            <div class="upload-form">
                <input type="file" id="imageFile" accept="image/*" onchange="previewImage(event)">
                <button onclick="analyzeUploadedImage()">이미지 분석</button>
            </div>
            <img id="preview" class="preview-image" alt="미리보기">
        </div>
        <div id="status"></div>
        <div id="analysis"></div>
        <div id="processed-image" style="margin: 20px 0; text-align: center;">
            <img id="contour-result" style="max-width: 100%; display: none;" alt="윤곽선 검출 결과">
        </div>
    </div>

    <script>
        function updateStatus(message, isError = false) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = isError ? 'error' : 'success';
        }

        function updateAnalysis(text) {
            const analysisDiv = document.getElementById('analysis');
            analysisDiv.textContent = text;
        }

        function previewImage(event) {
            const preview = document.getElementById('preview');
            const file = event.target.files[0];
            
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                }
                reader.readAsDataURL(file);
            } else {
                preview.style.display = 'none';
            }
        }

        async function connectDrone() {
            try {
                updateStatus("드론 연결 중...");
                const response = await fetch('/connect', {
                    method: 'POST'
                });
                const data = await response.json();
                updateStatus(data.message, data.status === 'error');
            } catch (error) {
                updateStatus('연결 중 오류가 발생했습니다: ' + error, true);
            }
        }

        async function scanSurroundings() {
            try {
                updateStatus("주변 스캔 중...");
                const response = await fetch('/scan', {
                    method: 'POST'
                });
                const data = await response.json();
                updateStatus(data.message, data.status === 'error');
                if (data.analysis) {
                    updateAnalysis(data.analysis);
                }
            } catch (error) {
                updateStatus('스캔 중 오류가 발생했습니다: ' + error, true);
            }
        }

        async function createPanorama() {
            try {
                updateStatus("파노라마 촬영 중...");
                const response = await fetch('/panorama', {
                    method: 'POST'
                });
                const data = await response.json();
                updateStatus(data.message, data.status === 'error');
                if (data.analysis) {
                    updateAnalysis(data.analysis);
                }
            } catch (error) {
                updateStatus('파노라마 촬영 중 오류가 발생했습니다: ' + error, true);
            }
        }

        async function analyzeUploadedImage() {
            const fileInput = document.getElementById('imageFile');
            const file = fileInput.files[0];
            
            if (!file) {
                updateStatus('파일을 선택해주세요.', true);
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            try {
                updateStatus("이미지 분석 중...");
                const response = await fetch('/analyze_uploaded', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                updateStatus(data.message, data.status === 'error');
                if (data.analysis) {
                    updateAnalysis(data.analysis);
                }
            } catch (error) {
                updateStatus('이미지 분석 중 오류가 발생했습니다: ' + error, true);
            }
        }

        async function controlDrone(command, parameters = {}) {
            try {
                updateStatus("드론 명령 실행 중...");
                const response = await fetch('/control', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        command: command,
                        parameters: parameters
                    })
                });
                const data = await response.json();
                updateStatus(data.message, data.status === 'error');
            } catch (error) {
                updateStatus('드론 제어 중 오류가 발생했습니다: ' + error, true);
            }
        }

        async function processImage() {
            try {
                updateStatus("이미지 처리 중...");
                const response = await fetch('/process_image', {
                    method: 'POST'
                });
                const data = await response.json();
                updateStatus(data.message, data.status === 'error');
                
                if (data.status === 'success' && data.image_url) {
                    const resultImage = document.getElementById('contour-result');
                    resultImage.src = data.image_url;
                    resultImage.style.display = 'block';
                }
            } catch (error) {
                updateStatus('이미지 처리 중 오류가 발생했습니다: ' + error, true);
            }
        }
    </script>
</body>
</html>
"""
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

if __name__ == '__main__':
    # 템플릿 생성
    ensure_template_exists()
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=5000) 