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

        # 안정화 체크
        self.candidate_move = None
        self.stable_count = 0
        
        # 안전장치: 마우스가 화면 구석으로 가면 프로그램 강제 종료
        pag.FAILSAFE = True 

    def click_event(self, event, x, y, flags, params):
        # 캘리브레이션을 위한 마우스 클릭 콜백
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.board_corners) < 4:
                self.board_corners.append((x, y))
                print(f"웹캠 좌표 설정: {x}, {y}")

    def calibrate_board(self):
        print("=== 1단계: 웹캠 체스판 캘리브레이션 ===")
        print("순서 중요: [1.좌상] -> [2.우상] -> [3.우하] -> [4.좌하]\n본인이 하는 색이 아래에 오도록 찍으세요.")

        time.sleep(1.0) 

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

            # 상태 표시
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
        print("\n=== 2단계: 모니터(웹 브라우저) 좌표 설정 ===")
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

    """
    def detect_move(self, current_frame):
        # 1. 이미지 전처리
        curr_warped = self.get_warped_frame(current_frame)
        curr_gray = cv2.cvtColor(curr_warped, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return None, None

        # 2. 차이 계산
        diff = cv2.absdiff(self.prev_gray, curr_gray)
        
        # [방어 1] 임계값 상향 (30)
        # 그림자 같은 옅은 변화(회색)는 0(검정)으로 만들어버리고, 확실한 변화만 255(흰색)로 만듦
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # 3. 변화된 칸 분석
        square_changes = []
        step = self.board_size // 8
        total_pixels_per_square = step * step

        for row in range(8):
            for col in range(8):
                x1, y1 = col * step, row * step
                x2, y2 = (col + 1) * step, (row + 1) * step
                
                roi = thresh[y1:y2, x1:x2]
                change_count = cv2.countNonZero(roi)
                
                # [방어 2] 면적 필터링
                # 한 칸 면적의 25% 이상이 변해야 인정 (작은 노이즈/구석 그림자 무시)
                if change_count > (total_pixels_per_square * 0.25): 
                    # 좌표 변환
                    sq_name = self.get_square_from_rect(x1 + step//2, y1 + step//2)
                    square_changes.append((change_count, sq_name))

        # 변화량이 큰 순서대로 정렬
        square_changes.sort(key=lambda x: x[0], reverse=True)

        # [방어 3] 조명 변화 감지 (Global Lighting Change)
        # 갑자기 4칸 이상이 동시에 변했다? 이건 기물 이동이 아니라 그림자/조명 문제임.
        # 3칸 변화는 앙파상일 수 있음!! << 나중에 적용
        if len(square_changes) > 3:
            print(f"⚠️ 조명/그림자 흔들림 감지됨 (변화된 칸 {len(square_changes)}개). 무시합니다.")
            return None, curr_gray

        # 4. 논리적 추론 (AI 대체 가능)
        # 상위 2개(가장 많이 변한 칸)만 가지고 판단
        if len(square_changes) >= 2:
            sq1 = square_changes[0][1]
            sq2 = square_changes[1][1]
            
            # log 확인용
            # print(f"감지 후보: {sq1}, {sq2} (변화량: {square_changes[0][0]}, {square_changes[1][0]})")
            # 흑일때 좌표를 이미 구체화했으므로 그대로 사용
            move1 = chess.Move.from_uci(sq1 + sq2)
            move2 = chess.Move.from_uci(sq2 + sq1)

            final_move = None
            if move1 in self.board.legal_moves:
                final_move = move1
            elif move2 in self.board.legal_moves:
                final_move = move2
            
            # 유효한 이동이 확인되면
            if final_move:
                print(f"✅ 이동 확정: {final_move}")
                self.board.push(final_move) # 내부 체스판 업데이트
                self.prev_gray = curr_gray # [중요] 이동이 성공했을 때만 기준 화면 업데이트!
                moved_color =  not self.board.turn
                return final_move.uci(), moved_color
        
        return None, None
    """
    
    def scan_current_view(self, current_frame):
        # 1. 전처리
        curr_warped = self.get_warped_frame(current_frame)
        curr_gray = cv2.cvtColor(curr_warped, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return None, curr_gray # 초기화용

        # 2. 차이 계산 (민감도 완화: 30 -> 50)
        diff = cv2.absdiff(self.prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        # 3. 변화된 칸 찾기
        square_changes = []
        step = self.board_size // 8
        total_pixels = step * step
        
        for row in range(8):
            for col in range(8):
                x1, y1 = col * step, row * step
                x2, y2 = (col + 1) * step, (row + 1) * step
                
                roi = thresh[y1:y2, x1:x2]
                # [수정] 한 칸의 15% 이상 변해야 인정 (노이즈 방어 강화)
                if cv2.countNonZero(roi) > (total_pixels * 0.15):
                    sq_name = self.get_square_from_rect(x1 + step//2, y1 + step//2)
                    square_changes.append((cv2.countNonZero(roi), sq_name))

        square_changes.sort(key=lambda x: x[0], reverse=True)

        # 그림자 등으로 너무 많이 변하면 무시
        if len(square_changes) > 4: 
            return None, curr_gray

        # 변화가 유의미한 2칸이 감지되면 문자열(예: "e2e4") 리턴
        if len(square_changes) >= 2:
            sq1 = square_changes[0][1]
            sq2 = square_changes[1][1]
            return sq1 + sq2, curr_gray
        
        return None, curr_gray

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
        
        # 마우스 원위치 (방해 안되게)
        pag.moveTo(10, 10)

"""
# === 메인 실행부 ===
if __name__ == "__main__":
    game = PhysicalChessBoard()
    
    # 플레이어 색상 선택
    while True:
        user_input = input("당신은 백(w)입니까 흑(b)입니까? (w/b): ").lower()
        if user_input == 'w':
            game.my_color = chess.WHITE
            print("⚪️ 당신은 [백(White)]입니다. 카메라 아래쪽이 1랭크입니다.")
            break
        elif user_input == 'b':
            game.my_color = chess.BLACK
            print("⚫️ 당신은 [흑(Black)]입니다. 카메라 아래쪽이 8랭크입니다.")
            break

    game.calibrate_board()  # 웹캠 설정
    game.calibrate_screen() # 화면 마우스 설정

    print("\n=== 게임 시작 ===")
    
    stable_counter = 0
    last_move_time = time.time()

    while True:
        ret, frame = game.cap.read()
        if not ret: break

        # 화면에 현재 상황 표시
        warped = game.get_warped_frame(frame)
        cv2.imshow("Original", frame)
        cv2.imshow("Warped View", warped)

        # 안정화 감지 (손이 움직이는 동안은 판독 X)
        # 현재 프레임과 이전 기준 프레임의 차이가 적을 때(수를 둔 후 손이 빠졌을 때) 로직 실행
        if game.prev_gray is None:
            curr_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            game.prev_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
            continue
            
        # 1초마다 한 번씩만 판독 시도 (과도한 연산 방지)
        if time.time() - last_move_time > 1.0:
            move_str, moved_color = game.detect_move(frame)
            if move_str:
                last_move_time = time.time() # 타이머 리셋

                if moved_color == game.my_color:    # 내 차례일 때만 마우스 조작
                    game.execute_on_screen(move_str)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    game.cap.release()
    cv2.destroyAllWindows()
"""

if __name__ == "__main__":
    game = PhysicalChessBoard()
    
    # 1. 색상 선택
    while True:
        user_input = input("당신은 백(w)입니까 흑(b)입니까? (w/b): ").lower()
        if user_input == 'w':
            game.my_color = chess.WHITE
            print("⚪️ 설정: 백(White).")
            break
        elif user_input == 'b':
            game.my_color = chess.BLACK
            print("⚫️ 설정: 흑(Black).")
            break

    game.calibrate_board()
    game.calibrate_screen()

    print("\n=== 게임 시작 ===")
    print("💡 팁: 기물을 옮기고 손을 확실히 치우세요.")
    
    while True:
        ret, frame = game.cap.read()
        if not ret: break

        warped = game.get_warped_frame(frame)
        cv2.imshow("Original", frame)
        
        # [디버깅용] 컴퓨터가 보는 흑백 화면 띄우기 (그림자 확인용)
        if game.prev_gray is not None:
             # 현재 화면과 기준 화면의 차이를 눈으로 보여줌 (흰색으로 번쩍이면 감지된 것)
             curr_gray_temp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
             curr_gray_temp = cv2.GaussianBlur(curr_gray_temp, (5, 5), 0)
             diff_debug = cv2.absdiff(game.prev_gray, curr_gray_temp)
             _, thresh_debug = cv2.threshold(diff_debug, 25, 255, cv2.THRESH_BINARY)    # 임계값 25
             cv2.imshow("Debug View (Threshold)", thresh_debug)
        else:
             cv2.imshow("Debug View (Threshold)", warped)

        # 1. 현재 화면 스캔
        detected_str, current_gray_img = game.scan_current_view(frame)

        # 2. 안정화 로직 (Debouncing)
        if detected_str:
            if detected_str == game.candidate_move:
                game.stable_count += 1
            else:
                game.candidate_move = detected_str
                game.stable_count = 1
                print(f"👀 감지 중... {detected_str}") # 흔들릴 때마다 출력됨
        else:
            game.stable_count = 0 # 변화가 사라지면 리셋
            
        # 3. 5프레임 연속으로 똑같은 수가 감지되면 -> "진짜 이동"으로 판정
        if game.stable_count >= 5:
            sq1, sq2 = game.candidate_move[:2], game.candidate_move[2:4]
            
            # 순서 조합 (e2->e4 인지 e4->e2 인지 확인)
            move1 = chess.Move.from_uci(sq1 + sq2)
            move2 = chess.Move.from_uci(sq2 + sq1)
            
            final_move = None
            if move1 in game.board.legal_moves:
                final_move = move1
            elif move2 in game.board.legal_moves:
                final_move = move2
            
            if final_move:
                print(f"\n✅ [이동 확정] {final_move.uci()}") # 이게 떠야 진짜 반영된 것임
                
                # 내부 보드 업데이트
                game.board.push(final_move)
                game.prev_gray = current_gray_img # 기준 화면 업데이트 (중요!)
                
                # 누구 턴이었는지 확인 (방금 둔 사람)
                moved_color = not game.board.turn 
                
                if moved_color == game.my_color:
                    print(f"   -> 내 턴이므로 마우스 클릭 실행")
                    game.execute_on_screen(final_move.uci())
                else:
                    print(f"   -> 상대 턴이므로 내부 상태만 동기화함")
                
                # 처리 후 초기화
                game.candidate_move = None
                game.stable_count = 0
                time.sleep(1.0) # 수 두고 나서 1초간 휴식 (중복 입력 방지)
            
            else:
                # 감지는 됐는데 규칙상 불가능한 수일 때
                if game.stable_count == 5: # 로그 한 번만 출력
                    print(f"❌ 규칙 위반 또는 불가능한 이동: {sq1} <-> {sq2}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    game.cap.release()
    cv2.destroyAllWindows()