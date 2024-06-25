Process for calculating angle of person relative to camera:
1. Detect key points on face -> ears, eyes, nose
2. Generate a target point (around face region). Ultimate goal will be for camera to aim directly at this point. 
3. Since distance between facial features remain relevantly constant across different people, pixel 
   distance between these keypoints are used to calculate the real distance (mm) that a pixel correlates to.
   Different facial features are picked when head is at different angles, depending on which ones are visible to the camera. 
4. Calculate real height of person based on bounding box height and pixl-real distance scale
5. Using equation distance= (focal length * real height of object  * image height) / (height of object in pixels × sensor height)
   determine the distance from the camera to the object, as if the object is shifted to be centered in the frame without, changing 
   its actual distance from the camera
6. Calc real horizontal distance from center of frame (AKA where camera currently aimed) to the target point
7. With arctan, determine horizontal angle deviation from where camera currently aimed to target point
8. Same process as 6-7, determine vertical angle deviation from where camera currently aimed to target point
9. According to horizontal and vertical angles of deviation, mechanicaly move camera to aim at target point

NOTE: this measurement process depends heavily on the person's face in the frame. Theoreticaly, if the camera shifts constantly to 
point at face region (where target point should be), then camera should always have person's face in frame

Method for tracking person keypoints:
YOLOv8 pose model, using default model (not custom trained). Performed int8 quantization with coco8 calibration dataset on 
model to run 3X faster, for real time frame processing. 

Camera:
Using Raspberry Pi 4 Model B, camera module v1. Send frames from Rasperry Pi (client) to computer(server) for processing
via TCP/IP connection over WiFi. Although Pi and Computer have same amount of RAM, computer has more storage
space for installing ultralytics. 

