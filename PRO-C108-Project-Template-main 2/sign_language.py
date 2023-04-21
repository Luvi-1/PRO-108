import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # accessing the landmarks by their position
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # code to detect the hand gestures
            finger_fold_status = []
            for tip in [4, 8, 12, 16, 20]:
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                cv2.circle(img, (x, y), 10, (0, 255, 0), -1)

                # Check if the finger is folded or not
                if lm_list[tip].x < lm_list[tip - 2].x:
                    cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            # Check if all fingers are folded and the thumb is up or down
            if all(finger_fold_status):
                if lm_list[3].y < lm_list[2].y < lm_list[1].y < lm_list[0].y:
                    print("LIKE")
                    cv2.putText(img, "LIKE!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif lm_list[3].y > lm_list[2].y > lm_list[1].y > lm_list[0].y:
                    print("DISLIKE")
                    cv2.putText(img, "DISLIKE!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 2, 2))
    

    cv2.imshow("hand tracking", img)
    cv2.waitKey(1)