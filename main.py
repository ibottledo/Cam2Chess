import cv2
import numpy as np
import pyautogui as pag
import time
import chess

class PhysicalChessBoard:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 웹캠 번호 (보통 0 또는 1)
        self.board_corners = []         # 웹캠상 체스판 네 모서리
        self.screen_corners = []        # 모니터상 체스판 좌표 (좌상단, 우하단)
        self.M = None                   # 투시 변환 행렬
        self.prev_gray = None           # 이전 안정된 상태의 흑백 이미지
        self.board = chess.Board()      # 내부 논리용 체스판 (규칙 검증용)
        self.board_size = 400           # 변환 후 이미지 크기 (400x400 픽셀)
        self.my_color = chess.WHITE     # 플레이어 색상 기본값
        self.candidate_move = None      # 감지된 후보 이동
        self.stable_count = 0           # 안정적으로 감지된 프레임 수
        self.no_move_start_time = None  # Auto-Healing용 타이머
        
        # 안전장치: 마우스가 화면 구석으로 가면 프로그램 강제 종료
        pag.FAILSAFE = True 

    def click_event(self, event, x, y, flags, params):
        # 캘리브레이션을 위한 마우스 클릭 콜백
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.board_corners) < 4:
                self.board_corners.append((x, y))
                print(f"웹캠 좌표 설정: {x}, {y}")

    def calibrate_board(self):
        print("===웹캠 체스판 캘리브레이션 ===")
        print("순서 중요: [1.좌상] -> [2.우상] -> [3.우하] -> [4.좌하]\n본인이 플레이하는 기물이 아래에 오도록 찍으세요.")

        # time.sleep(1.0) 

        if not self.cap.isOpened():
            print("❌ 카메라 연결 실패.")
            exit()

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.click_event)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            pts_cnt = len(self.board_corners)
            
            # 찍은 점 선으로 잇기
            if pts_cnt > 0:
                # 0->1, 1->2, 2->3 순서로 선 그리기
                for i in range(pts_cnt - 1):
                    cv2.line(frame, self.board_corners[i], self.board_corners[i+1], (0, 255, 0), 2)
                
                # 4개를 다 찍었으면 마지막 점(3)과 첫 점(0)도 이어서 사각형 완성
                if pts_cnt == 4:
                    cv2.line(frame, self.board_corners[3], self.board_corners[0], (0, 255, 0), 2)

            # 카메라 화면 상단에 상태 표시
            status_text = f"Points: {pts_cnt} / 4"
            if pts_cnt == 4: status_text = "Done!"
                
            cv2.rectangle(frame, (5, 5), (250, 45), (0,0,0), -1) 
            cv2.putText(frame, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            for i, pt in enumerate(self.board_corners):
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                cv2.putText(frame, str(i+1), (pt[0]+10, pt[1]+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Calibration", frame)
            
            if pts_cnt == 4:
                cv2.waitKey(1000) # 확인하도록 1초 보여줌
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): exit()
            elif key == ord('r'):
                self.board_corners = []
                print("🔄 리셋")
        
        cv2.destroyWindow("Calibration")

        # 변환 행렬 계산
        pts1 = np.float32(self.board_corners)
        pts2 = np.float32([[0, 0], [self.board_size, 0], 
                           [self.board_size, self.board_size], [0, self.board_size]])
        self.M = cv2.getPerspectiveTransform(pts1, pts2)
        print("✅ 캘리브레이션 완료!")

    def calibrate_screen(self):
        print("\n=== 모니터(웹 브라우저) 좌표 설정 ===")
        print("마우스를 모니터의 chess.com 보드 '좌상단(a8) 모서리'에 두고 Enter를 치세요.")
        input("준비되면 Enter...")
        x1, y1 = pag.position()
        print(f"좌상단 저장됨: {x1}, {y1}")
        
        print("마우스를 모니터의 chess.com 보드 '우하단(h1) 모서리'에 두고 Enter를 치세요.")
        input("준비되면 Enter...")
        x2, y2 = pag.position()
        print(f"우하단 저장됨: {x2}, {y2}")
        
        self.screen_corners = [(x1, y1), (x2, y2)]
        print("화면 좌표 설정 완료!")

    def get_warped_frame(self, frame):
        # 캘리브레이션된 정보로 체스판을 정사각 평면으로 폄
        return cv2.warpPerspective(frame, self.M, (self.board_size, self.board_size))

    def get_square_from_rect(self, x, y):
        # 400x400 이미지를 8x8로 나누어 체스 좌표(a1~h8) 반환
        # 내가 흑인 경우 보드 좌표계 반전
        col = x // (self.board_size // 8)
        row = 7 - (y // (self.board_size // 8)) # 0이 8랭크(위쪽)이므로 반전 필요
        
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        if self.my_color == chess.BLACK:
            col = 7 - col
            row = 7 - row
        return files[col] + str(row + 1)
    
    # Move, Gray_Image, Total_Diff반환
    def scan_current_view(self, current_frame):
        curr_warped = self.get_warped_frame(current_frame)          # 체스판 부분만 추출
        curr_gray = cv2.cvtColor(curr_warped, cv2.COLOR_BGR2GRAY)   # 흑백 변환
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)          # 블러로 노이즈 완화

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return None, curr_gray, 0

        # 이전 화면과 차이 계산 (민감도: 50)
        diff = cv2.absdiff(self.prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        # 전체 변화량(노이즈) 측정
        total_diff_pixels = cv2.countNonZero(thresh)

        # 변화된 칸 찾기
        square_changes = []
        step = self.board_size // 8
        total_pixels = step * step
        
        for row in range(8):
            for col in range(8):
                x1, y1 = col * step, row * step
                x2, y2 = (col + 1) * step, (row + 1) * step
                
                roi = thresh[y1:y2, x1:x2]
                # 한 칸의 20% 이상 변해야 인정 (노이즈 방어 강화)
                if cv2.countNonZero(roi) > (total_pixels * 0.20):
                    sq_name = self.get_square_from_rect(x1 + step//2, y1 + step//2)
                    square_changes.append((cv2.countNonZero(roi), sq_name))

        # 변화량이 큰 순서대로 정렬
        square_changes.sort(key=lambda x: x[0], reverse=True)

        # 그림자 등으로 너무 많이 변하면 무시
        # 5프레임 유지 조건을 넣었으니 완화해서 8칸으로 변경
        if len(square_changes) > 8: 
            return None, curr_gray, total_diff_pixels

        # 변화가 유의미한 2칸이 감지되면 문자열(예: "e2e4") 리턴
        if len(square_changes) >= 2:
            sq1 = square_changes[0][1]
            sq2 = square_changes[1][1]
            return sq1 + sq2, curr_gray, total_diff_pixels
        
        return None, curr_gray, total_diff_pixels
    
    def force_reset_background(self, frame):
        print("🔄 [시스템] 배경 기준점 재설정 (Recalibrating Background...)")
        warped = self.get_warped_frame(frame)
        curr_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        self.prev_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        self.candidate_move = None
        self.stable_count = 0
        self.no_move_start_time = None

    def execute_on_screen(self, move_string):
        if not self.screen_corners: return

        # move_string 예: "e2e4"
        start_sq, end_sq = move_string[:2], move_string[2:4]
        files = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
        
        def get_pos(sq):
            f = files[sq[0]]
            r = int(sq[1])
            
            x1, y1 = self.screen_corners[0]
            x2, y2 = self.screen_corners[1]
            w, h = x2 - x1, y2 - y1
            
            # 백: a1이 좌하단 / 흑: h8이 좌하단 (화면이 뒤집힘)
            if self.my_color == chess.WHITE:
                target_x = x1 + (f * w/8) + w/16
                target_y = y1 + ((8-r) * h/8) + h/16
            else:
                # 흑일 때 화면 좌표 계산 (좌우, 상하 반전)
                target_x = x1 + ((7-f) * w/8) + w/16
                target_y = y1 + ((r-1) * h/8) + h/16
                
            return target_x, target_y

        # 마우스 조작
        sx, sy = get_pos(start_sq)
        ex, ey = get_pos(end_sq)

        pag.click(sx, sy) # 출발지 클릭
        time.sleep(0.1)
        pag.click(ex, ey) # 도착지 클릭
        
        # 마우스 원위치
        pag.moveTo(10, 10)

if __name__ == "__main__":
    game = PhysicalChessBoard()
    
    # 색상 선택
    while True:
        user_input = input("당신은 백(w)입니까 흑(b)입니까? (w/b): ").lower()
        if user_input == 'w':
            game.my_color = chess.WHITE
            break
        elif user_input == 'b':
            game.my_color = chess.BLACK
            break

    game.calibrate_board()
    game.calibrate_screen()

    print("\n=== 게임 시작 ===")
    print("💡 팁: 'u' 키를 누르면 강제로 배경을 리셋합니다.")
    
    # 정지 상태 확인을 위한 변수, gray 이미지 저장
    last_loop_gray = None 

    while True:
        ret, frame = game.cap.read()
        if not ret: break

        warped = game.get_warped_frame(frame)
        cv2.imshow("Original", frame)
        
        # 현재 상태 스캔 (기준 배경과의 차이)
        detected_str, current_gray_img, diff_from_bg = game.scan_current_view(frame)

        # (디버깅용) 변화된 부분 시각화
        if game.prev_gray is not None:
             # 현재 화면과 기준 화면의 차이를 구해서 보여줌
             diff_debug = cv2.absdiff(game.prev_gray, current_gray_img)
             _, thresh_debug = cv2.threshold(diff_debug, 30, 255, cv2.THRESH_BINARY)
             cv2.imshow("Debug View", thresh_debug) # <--- 이 창이 뜹니다

        # 화면이 정지해 있는지 확인 (프레임 간 차이)
        is_static = False
        if last_loop_gray is not None:
            # 바로 직전 프레임과 현재 프레임 비교
            frame_diff = cv2.absdiff(last_loop_gray, current_gray_img)
            _, frame_thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            inter_frame_change = cv2.countNonZero(frame_thresh)
            
            # 변화된 픽셀이 200개 미만이면 "화면이 멈춰있다"고 판단
            if inter_frame_change < 200:
                is_static = True
            
            # (디버깅용) 얼마나 움직이는지 출력
            # print(f"움직임: {inter_frame_change}, 배경오차: {diff_from_bg}")

        last_loop_gray = current_gray_img.copy() # 다음 비교를 위해 저장

        # ======================================================
        # 로직 시작
        # ======================================================

        if detected_str:
            # A. 유효한 이동(Move) 감지됨 -> 정상 게임 진행
            game.no_move_start_time = None 

            if detected_str == game.candidate_move:
                game.stable_count += 1
            else:
                game.candidate_move = detected_str
                game.stable_count = 1
                print(f"👀 감지 중... {detected_str}")

            if game.stable_count >= 5:
                sq1, sq2 = game.candidate_move[:2], game.candidate_move[2:4]
                move1 = chess.Move.from_uci(sq1 + sq2)
                move2 = chess.Move.from_uci(sq2 + sq1)
                
                final_move = None
                if move1 in game.board.legal_moves: final_move = move1
                elif move2 in game.board.legal_moves: final_move = move2
                
                if final_move:
                    print(f"\n✅ [이동 확정] {final_move.uci()}")
                    game.board.push(final_move)
                    game.prev_gray = current_gray_img 
                    
                    if game.board.turn != game.my_color: # 내 턴 끝남
                        game.execute_on_screen(final_move.uci())
                    else:
                        print(f"[상대 수] 내부 보드만 동기화")
                    
                    game.candidate_move = None
                    game.stable_count = 0
                    time.sleep(0.5) # 수 두고 잠깐 대기
        
        else:
            # B. 감지된 이동 없음
            game.stable_count = 0
            
            # Auto-Healing 로직
            # 조건 1: 배경과 다름 (diff_from_bg > 500) -> 뭔가 잘못됨
            # 조건 2: 근데 화면은 안 움직임 (is_static) -> 손이 아니라 배경이 틀어진 것
            
            if (diff_from_bg > 500) and is_static:
                
                if game.no_move_start_time is None:
                    game.no_move_start_time = time.time()
                    # print("⏳ 정적 노이즈 감지... 타이머 시작")
                
                elif time.time() - game.no_move_start_time > 2.5:
                    print(f"⚠️ [자동 보정] 화면이 정지된 상태로 틀어져 있음 -> 기준점 갱신")
                    game.prev_gray = current_gray_img
                    game.no_move_start_time = None
            
            else:
                # 화면이 흔들리거나(움직임), 깨끗하면 타이머 리셋
                if game.no_move_start_time is not None:
                    # print("움직임 감지됨/화면 복구됨 -> 타이머 리셋")
                    game.no_move_start_time = None

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('u'): game.force_reset_background(frame)

    game.cap.release()
    cv2.destroyAllWindows()