import cv2
import mediapipe as mp
import numpy as np
import time

class Joint:
    def __init__(self, name, threshold_small, threshold_large, small_prompt=None, large_prompt=None):
        self.name = name
        self.threshold_small = threshold_small
        self.threshold_large = threshold_large
        self.angle = 0
        self.coords = []
        self.small_prompt = small_prompt or f"Straighten your {self.name}"
        self.large_prompt = large_prompt or small_prompt or f"Bend your {self.name}"

    def calculate_angle(self, prev_joint, next_joint):
        a = np.array(prev_joint)
        b = np.array(self.coords)
        c = np.array(next_joint)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        self.angle = np.abs(radians * 180.0 / np.pi)
        if self.angle > 180.0:
            self.angle = 360 - self.angle

    def get_feedback_color(self):
        if self.angle < self.threshold_small:
            return (0, 0, 255)
        elif self.angle > self.threshold_large:
            return (255, 0, 0)
        else:
            return (0, 255, 0)

    def get_prompt(self):
        if self.angle < self.threshold_small:
            return self.small_prompt
        elif self.angle > self.threshold_large:
            return self.large_prompt
        return None

# Define the joints with angle thresholds specific to Half Standing Fold Pose
joints = [
    Joint('back', 80, 100, "Keep your back straight and parallel to the floor."),
    Joint('hips', 80, 100, "Bend at the hips to achieve a right angle."),
    Joint('knees', 165, 180, "Keep your knees straight.")
]

last_check_time = 0
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
res_w = 640
res_h = 480
cap.set(3, res_w)
cap.set(4, res_h)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            current_time = time.time()

            if current_time - last_check_time >= 1:
                # Update joint coordinates
                joints[0].coords = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                ]
                joints[1].coords = [
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                ]
                joints[2].coords = [
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                ]

                # Calculate angles for each joint
                # Back (elbow-shoulder-hip for alignment with floor)
                joints[0].calculate_angle(joints[1].coords, [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])
                
                # Hips (alignment of spine to legs)
                joints[1].calculate_angle(joints[2].coords, joints[0].coords)

                # Knees (to check if they are straight)
                joints[2].calculate_angle(joints[1].coords, [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])

                # Prompts for the user
                prompt = None
                for joint in joints:
                    prompt = joint.get_prompt()
                    if prompt:
                        break
                
                if prompt:
                    print(prompt)
                else:
                    print("Correct")

                last_check_time = current_time

            # Display angles with respective color
            # for joint in joints:
            #     position = tuple(np.multiply(joint.coords, [res_w, res_h]).astype(int))
            #     cv2.putText(image, str(int(joint.angle)),
            #                 position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                 joint.get_feedback_color(), 2, cv2.LINE_AA)

        except:
            pass

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
