import cv2
import mediapipe as mp
import numpy as np
import time
import math

last_check_time = 0
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
res_w = 1280
res_h = 960
cap.set(3, res_w)
cap.set(4, res_h)
global_start_sign = 0


def starting_condition(landmarks):
    left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
    right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
    shoulder_height = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

    # print(f"left x = {left_index[0]} y = {left_index[1]}")
    # print(f"rit x = {right_index[0]} y = {right_index[1]}")
    # print(f"shoulder ht = {shoulder_height}")

    if left_index[1] > shoulder_height and right_index[1] > shoulder_height:
        distance = math.sqrt((right_index[0] - left_index[0]) ** 2 + (right_index[1] - left_index[1]) ** 2)
        distance = distance * 100
        if distance < 3:
            return 1


def termination_condition(landmarks):
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    waist_height = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y

    # print(f"left index x = {left_wrist[0]} y = {left_wrist[1]}")
    # print(f"rit index x = {right_wrist[0]} y = {right_wrist[1]}")
    # print(f"left elbow x = {left_elbow[0]} y = {left_elbow[1]}")
    # print(f"rit elbow x = {right_elbow[0]} y = {right_elbow[1]}")
    # print(f"waist ht = {waist_height}")

    if left_elbow[1] < waist_height and right_elbow[1] < waist_height and left_wrist[1] < left_elbow[1] and right_wrist[1] < right_elbow[1]:
        if left_wrist[0] < right_wrist[0] and right_elbow[0] < left_elbow[0]:
            return True


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            start_flag = starting_condition(landmarks)
            if start_flag == 1: global_start_sign = 1

            if global_start_sign == 1:
                if termination_condition(landmarks) == True:
                    cap.release()
                    cv2.destroyAllWindows()

                cv2.putText(image, "ACTIVE",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                            )

                current_time = time.time()
                if current_time - last_check_time >= 1:
                    # Considering left-side features for analysing posture
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    waist = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # Calculating angle for various joints
                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    shoulder_angle = calculate_angle(elbow, shoulder, waist)
                    waist_angle = calculate_angle(shoulder, waist, knee)
                    knee_angle = calculate_angle(waist, knee, ankle)

                    # Prompts for the user
                    if elbow_angle < 165:
                        elbow_prompt = "Kindly straighten your elbows"
                        print(elbow_prompt)

                    if (shoulder_angle > 200 or shoulder_angle < 160):
                        shoulder_prompt = "Your torso and arm should be held in a straight line. Kindly straighten your shoulder."
                        print(shoulder_prompt)

                    if waist_angle > 75:
                        waist_prompt = "Further close the gap between your arms and legs."
                        print(waist_prompt)

                    if knee_angle < 160:
                        knee_prompt = "Your knee should be straight."
                        print(knee_prompt)

                    last_check_time = current_time

            # Visualize angle
            # cv2.putText(image, str(angle),
            #             tuple(np.multiply(elbow, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #             )

        except:
            pass

        # Drawing landmarks on image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('AsanAI Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()