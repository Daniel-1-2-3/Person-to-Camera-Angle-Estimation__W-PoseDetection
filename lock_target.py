import cv2
from ultralytics import YOLO
import math
import numpy as np
import time
    
class Target:
    def __init__(self):
        #get YOLOv8 model
        self.yolo_model = YOLO('yolov8n-pose_int8_openvino_model/', task='pose')
        #ways to speed up inference: allow parallel calculations on multiple CPU cores, pruning(remove some node connections), layer fusion(combines convolutional and activation layer, which are suually seperate when generating feature maps)
        #chosen: converting the model to tensorRT to quantize it, turning all 32 decimal weights to 8 decimal
        
    def find_target_coordinates(self, frame):
        frame = cv2.flip(frame, 1)
        results = self.yolo_model(frame, conf=0.6, save=False)
        
        reference_points = [] #points to calculate distance that a pixel correlates to. This list includes important FACIAL keypoints (unlike keypoints list), such as eyes, ears and nose
        keypoints = [] #all the points detected, this is used to calculate target point (since need to take into account points on whole body to determine chest area) 
        keypoints_x = []
        keypoints_y = []
        aim_point = None #target point 
        
        if results[0].keypoints is not None:
            person = (results[0].keypoints)[0] #coordinates of keypoints for first person detected
            if person.conf is None:
                return frame, 'N/A', 'N/A'
            
            for i, (keypoint, conf) in enumerate(zip(person.xy[0].tolist(), person.conf[0])):
                x, y = keypoint
                if int(x)!=0 and int(y)!=0:
                    keypoints.append((int(x), int(y)))
                    keypoints_x.append(int(x))
                    keypoints_y.append(int(y))
                reference_points.append((int(x), int(y))) if conf>0.7 else reference_points.append((0, 0))
                
            if len(keypoints)!=0:
                
                aim_point_x = int(sum(keypoints_x[0:9])/len(keypoints_x[0:9]))
                aim_point_y = int(sum(keypoints_y[0:9])/len(keypoints_y[0:9])) + int(frame.shape[0]/20)
                aim_point = (aim_point_x, aim_point_y) #target point at chest 
                
                reference_points = reference_points[0:7] #refrence points for calculating the distance each pixel corresponds to 
                mm_per_pixel, head_orientation = self.calc_mm_per_pixel(reference_points) #if head_orientation is N/A, indicates not enough reference points located on face to properly perform disance calculations
                
                #draw bounding boxes
                boxes = results[0].boxes.xyxy.tolist()
                classes = []
                for class_id in results[0].boxes.cls.tolist():
                    classes.append((results[0].names)[class_id])
                for bbox in boxes:
                    frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(200, 200, 200))
                
                #calculate distance to person, if all displays N/A, indicates not enough keypoints available to effectively calculate distance that each pixel correlates to
                distance, real_height, angle_h, angle_v = self.calc_distance(frame, aim_point, boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3], mm_per_pixel)
                cv2.putText(frame, f'Aprox Distance: {str(distance)} m', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)
                cv2.putText(frame, f'Deviation from x-axis: {str(angle_v)} deg', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)
                cv2.putText(frame, f'Deviation from y-axis: {str(angle_h)} deg', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)
                cv2.putText(frame, f'Real Height: {real_height}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)
                cv2.putText(frame, f'Head Orientation {head_orientation}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)
                        
                cv2.circle(frame, (reference_points[0][0], reference_points[0][1]), 3, (255, 255, 255), -1)
                for i, (x, y) in enumerate(reference_points[1:]):
                    cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
                
                cv2.circle(frame, (aim_point), 6, (0, 0, 255), -1) #draw target point
                cv2.circle(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), 6, (0, 150, 0), -1) #center of frame, where cam is currently pointing 
                
                label = f'({aim_point_x},{aim_point_y})'
                cv2.putText(frame, label, (int(aim_point_x) + 10, int(aim_point_y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                return frame, angle_h, angle_v
        else:
            return frame, 'N/A', 'N/A'
    
    def calc_mm_per_pixel(self, ref_list):
        mm_per_pixel = 0 
        nose, right_eye, left_eye, right_ear, left_ear = ref_list[:5]
        orientation = 'N/A'
        
        #case facing forwards, #both ears, both eyes, nose are visible 
        if sum(right_ear)!=0 and sum(left_ear)!=0 and sum(left_eye)!=0 and sum(right_eye)!=0 and sum(nose)!=0: 
            eye_eye_pixeld =  abs(left_eye[0] - right_eye[0]) #real distance 63mm
            eye_nose_pixeld = abs(nose[1] - (left_eye[1] + right_eye[1])/2) #real distance 40mm
            mm_per_width_pixel = 63/eye_eye_pixeld
            mm_per_height_pixel = 30/eye_nose_pixeld
            mm_per_pixel = (mm_per_width_pixel + mm_per_height_pixel)/2 #average mm per width pixel and mm per height pixel due to account for slight margins of errors in keypoint estimation
            orientation = 'forwards'
            
        #case facing slightly left, left eye still visible
        elif sum(right_ear)!=0 and sum(left_eye)!=0 and sum(nose)!=0 and sum(right_eye)!=0 and sum(left_ear)==0: 
            ear_eye_pixeld = math.sqrt((right_ear[0] - right_eye[0])**2 + (right_ear[1] - right_eye[1])**2) #real distance 85mm
            eye_nose_pixeld = abs(nose[1] - right_eye[1]) #real distance 40mm)
            mm_per_width_pixel = 85/ear_eye_pixeld
            mm_per_height_pixel = 30/eye_nose_pixeld
            mm_per_pixel = (mm_per_width_pixel + mm_per_height_pixel)/2 
            orientation = 'slightly left'
        
        #case facing slightly right, right eye still visible 
        elif sum(left_ear)!=0 and sum(right_eye)!=0 and sum(nose)!=0 and sum(left_eye)!=0 and sum(right_ear)==0: 
            ear_eye_pixeld = math.sqrt((left_ear[0] - left_eye[0])**2 + (left_ear[1] - left_eye[1])**2) #real distance about 80mm 
            eye_nose_pixeld = abs(nose[1] - right_eye[1]) #real distance 40mm)
            mm_per_width_pixel = 80/ear_eye_pixeld
            mm_per_height_pixel = 30/eye_nose_pixeld
            mm_per_pixel = (mm_per_width_pixel + mm_per_height_pixel)/2 
            orientation = 'slightly right'
        
        #case facing hard left, left eye not visible 
        elif sum(right_ear)!=0 and sum(right_eye)!=0 and sum(left_ear)==0 and sum(left_eye)==0: 
            ear_eye_pixeld = math.sqrt((right_ear[0] - right_eye[0])**2 + (right_ear[1] - right_eye[1])**2) #real distance about 100mm, assuming head tilted about 120-130 degrees from forward pos 
            mm_per_pixel = 100/ear_eye_pixeld
            orientation = 'hard left'
        
        #case facing hard right, nose not visible 
        elif sum(left_ear)!=0 and sum(left_eye)!=0 and sum(right_ear)==0 and sum(right_eye)==0: 
            ear_eye_pixeld = math.sqrt((left_ear[0] - left_eye[0])**2 + (left_ear[1] - left_eye[1])**2) 
            mm_per_pixel = 100/ear_eye_pixeld
            orientation = 'hard right'
        
        #case facing backwards, only 2 ears visible
        elif sum(left_ear)!=0 and sum(right_ear)!=0 and sum(left_eye) + sum(right_eye) + sum(nose) == 0:
            ear_ear_pixeld = math.sqrt((left_ear[0] - right_ear[0])**2 + (left_ear[1] - right_ear[1])**2) #real distance about 130mm
            mm_per_pixel = 130/ear_ear_pixeld
            orientation = 'backwards'
            
        return mm_per_pixel, orientation
    
    #!!! current calculations assume camera principle axis is perpendicular to the object if object was centered
    def calc_distance(self, frame, aim_point, x, y, w, h, mm_per_pixel): #x, y, w, h are bounding box coordinates, top left and bottom right
        if mm_per_pixel == 0:
            return 'N/A', 'N/A', 'N/A', 'N/A' 
        
        #below parameters specified by raspberry pi documentation, cam module v1, OmniVision OV5647 sensor
        focal_length = 3.6
        image_height, image_width, _ = frame.shape
        sensor_height, sensor_width = (3.76, 2.74) #mm
        
        object_pixel_height = abs(y - h)
        object_pixel_width = abs(x - w)
        
        object_real_height = (object_pixel_height * mm_per_pixel) #mm
        object_real_width = (object_pixel_width * mm_per_pixel) #mm
        
        #calculate distance from camera to object, IF object was centered,
        d1 = (focal_length * object_real_height * image_height)/(object_pixel_height * sensor_height) #distance calculated from heights
        d2 = (focal_length * object_real_width * image_width)/(object_pixel_width * sensor_width) #distance calculated from widths
        distance_center = int(d1 + d2)/2 #mm, average distances calculated from height and width for increased accuracy, reduce sources of error
        
        #calculate distance from camera to object (aim_point), where object is off center
        center_x, center_y = image_width/2, image_height/2 #coordinates of the center of image
        aim_point_x, aim_point_y = aim_point
        angle_h = int(math.atan((center_x - aim_point_x) * mm_per_pixel/distance_center) * (180/math.pi)) #angle deviation from y-axis
        angle_v = int(math.atan((center_y - aim_point_y) * mm_per_pixel/distance_center) * (180/math.pi)) #angle deviation from x-axis
        
        return round(distance_center/1000, 2), round(object_real_height/1000, 2), angle_h, angle_v
    
if __name__ == "__main__":
    searcher = Target()
    vid = cv2.VideoCapture(0)  
    
    speeds = []
    
    while(True): 
        ret, frame = vid.read() 
        start_time = time.time()
        frame, angle_h, angle_v = searcher.find_target_coordinates(frame)
        end_time = time.time()
        speeds.append(end_time - start_time)
        cv2.imshow('frame', frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    vid.release() 
    cv2.destroyAllWindows() 
    
    print('Average inference time', sum(speeds)/len(speeds))
    
