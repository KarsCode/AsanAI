from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np

app = FastAPI()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Utility function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

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

    def get_prompt(self):
        if self.angle < self.threshold_small:
            return self.small_prompt
        elif self.angle > self.threshold_large:
            return self.large_prompt
        return None

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Server is running"}

# Pose estimation endpoint with poseType
@app.post("/pose-estimation/")
async def pose_estimation(file: UploadFile = File(...), poseType: str = Form(...)):
    contents = await file.read()
    print(poseType)

    # Check if the buffer is empty
    if not contents:
        return JSONResponse(content={"error": "Empty image buffer received"}, status_code=400)

    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Verify if frame was decoded successfully
    if frame is None:
        return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        try:
            landmarks = results.pose_landmarks.landmark
            # Define keypoints and calculate angles
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            waist = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = calculate_angle(elbow, shoulder, waist)
            waist_angle = calculate_angle(shoulder, waist, knee)
            knee_angle = calculate_angle(waist, knee, ankle)
            
            # Generate prompts based on poseType
            prompts = []
            if poseType.lower() == "dog":
                if elbow_angle < 165:
                    prompts.append("Kindly straighten your elbows")
                if shoulder_angle > 200 or shoulder_angle < 160:
                    prompts.append("Your torso and arm should be in a straight line. Kindly straighten your shoulder.")
                if waist_angle > 75:
                    prompts.append("Further close the gap between your arms and legs.")
                if knee_angle < 160:
                    prompts.append("Your knee should be straight.")
                if not prompts:
                    prompts.append("Correct alignment achieved for Downward Dog.")
            
            elif poseType.lower() == "tree":
                if knee_angle > 150 or knee_angle < 90:
                    prompts.append("The bent knee angle should be between 90 and 150 degrees.")
                if shoulder_angle < 160 or shoulder_angle > 200:
                    prompts.append("Your torso should be upright.")
                if elbow_angle > 170:
                    prompts.append("Keep your arms closer to your head.")
                if not prompts:
                    prompts.append("Correct alignment achieved for Tree Pose.")
            
            elif poseType.lower() == "half-standing fold":
                # Define the joints with angle thresholds specific to Half Standing Fold Pose
                joints = [
                    Joint('back', 80, 100, "Keep your back straight and parallel to the floor."),
                    Joint('hips', 80, 100, "Bend at the hips to achieve a right angle."),
                    Joint('knees', 165, 180, "Keep your knees straight.")
                ]
                
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

                # Generate prompts for each joint
                for joint in joints:
                    prompt = joint.get_prompt()
                    if prompt:
                        prompts.append(prompt)

                if not prompts:
                    prompts.append("Correct alignment achieved for Half Standing Fold Pose.")

            elif poseType.lower() == "mountain":
                # Define the joints with angle thresholds specific to Half Standing Fold Pose
                joints = [
                    Joint('elbow', 170, 180, "Keep your arms straight and aligned with your body."),
                    Joint('shoulder', 170, 180, "Align your shoulders vertically with your torso."),
                    Joint('waist', 170, 180, "Keep your torso upright."),
                    Joint('knee', 170, 180, "Keep your knees straight.")
                ]
                
                joints[0].coords = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                ]
                joints[1].coords = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                ]
                joints[2].coords = [
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                ]
                joints[3].coords = [
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                ]

                # Calculate angles for each joint
                joints[0].calculate_angle(joints[1].coords, [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                                                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
                joints[1].calculate_angle(joints[2].coords, joints[0].coords)
                joints[2].calculate_angle(joints[1].coords, joints[3].coords)
                joints[3].calculate_angle(joints[2].coords, [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                                                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])

                # Generate prompts for each joint
                for joint in joints:
                    prompt = joint.get_prompt()
                    if prompt:
                        prompts.append(prompt)

                if not prompts:
                    prompts.append("Correct alignment achieved for Mountain Pose.")
            
            return JSONResponse(content={"prompts": prompts})
        
        except Exception as e:
            return JSONResponse(content={"error": "Could not detect pose landmarks", "details": str(e)})

# To run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
