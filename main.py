# main.py에 대한 예시 파일 안내입니다!

# import torch
import random

model = None  # 전역 변수로 모델을 유지해주세요! 변수 이름도 model로 유지하셔야 합니다!

def init():
    # << 체점 시 양쪽 에이전트에 대해서 처음 한 번 실행되는 함수입니다. >>
    # 딥러닝을 통해 게임 에이전트 모델을 training하신 경우에는 모델을 델러오고, 평가 모드로 전환하는 부분을 이곳에 넣으셔야 합니다.
    # 딥러닝을 사용하지 않으셨더라도, Model-based AI로 에이전트를 만드신 분들도 이곳에서 모델/데이터 로딩을 하시면 됩니다.
    global model
    
    # 예시: 학습된 모델 로드
    # model = torch.load("weights.pt", map_location="cpu")
    # model.eval() # model을 training이 아닌 evalutation 모드로 전환

def run(board_lines, xsize, ysize):
    # << 에이전트의 차례가 될 때마다 실행되는 함수입니다. >>
    # 함수의 입력은 위와 같이 현재 board의 현재 상태 (놓인 수들)과 보드의 크기가 제공됩니다.
    # 이러한 입력 값을 바탕으로, 다음과 같이 놓을 수를 반환해주시면 됩니다.

    # x와 y는 선분이 시작되는 점의 좌표에 대한 x, y 성분입니다. 이 때, x가 가로, y가 세로를 의미합니다.
    # 좌상단 점에서부터 시작되며, x와 y는 각각 0 <= x <= xsize, 0 <= y <= ysiz는를 만족하는 정수입니다.
    # z는 선분을 그릴 방향이며 0이면 가로(오른쪽), 1이면 세로(아래)로 선분을 그리게 됨.
    # 이 때, x = xsize인 경우는 시작점이 가장 오른래에 있으므로 z = 0일 수 없으며,
    # 마찬가지로 y = ysize인 경우에는 시작점이 가장 아래 줄에 있으므로 z = 1일 수 없음.
    x = random.randint(0, xsize - 1)
    y = random.randint(0, ysize - 1)
    z = random.choice([0, 1])

    # 이렇게 구한 세 정수 x, y, z를 list로 묶어 반환하면 됨.
    return [x, y, z]