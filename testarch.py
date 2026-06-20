import cv2
import numpy as np
import random
import time

# ====== 状态定义 ======
STATES = ["neutral", "happy", "surprised", "sleepy"]

# ====== 像素脸库（简化版16x16）======
def make_face(state):
    canvas = np.zeros((16, 16), dtype=np.uint8)

    if state == "neutral":
        cv2.line(canvas, (4, 10), (12, 10), 255, 1)

    elif state == "happy":
        cv2.ellipse(canvas, (8, 10), (5, 3), 0, 0, 180, 255, 1)

    elif state == "surprised":
        cv2.circle(canvas, (6, 8), 1, 255, -1)
        cv2.circle(canvas, (10, 8), 1, 255, -1)
        cv2.circle(canvas, (8, 12), 2, 255, 1)

    elif state == "sleepy":
        cv2.line(canvas, (5, 8), (7, 9), 255, 1)
        cv2.line(canvas, (9, 9), (11, 8), 255, 1)

    return canvas


# ====== 模拟“表情识别” ======
def fake_state_from_camera(frame):
    # 后面这里会换成 EAR + 眉毛检测

    t = time.time()

    if int(t) % 8 < 2:
        return "happy"
    elif int(t) % 8 < 4:
        return "neutral"
    elif int(t) % 8 < 6:
        return "surprised"
    else:
        return "sleepy"


# ====== 主循环 ======
def main():
    cap = cv2.VideoCapture(0)

    last_state = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        state = fake_state_from_camera(frame)

        # 状态变化才打印
        if state != last_state:
            face = make_face(state)

            print("\n========== STATE CHANGE ==========")
            print("STATE:", state)
            print("PIXEL FACE:")
            print(face)

            last_state = state

        # 降低CPU
        time.sleep(0.1)

        if cv2.waitKey(1) == 27:
            break

    cap.release()


if __name__ == "__main__":
    main()
