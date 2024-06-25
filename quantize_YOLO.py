from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

#ways to speed up inference: allow parallel calculations on multiple CPU cores, pruning(remove some node connections), layer fusion(combines convolutional and activation layer, which are suually seperate when generating feature maps)
#chosen: converting the model to tensorRT to quantize it, turning all 32 decimal weights to 8 decimal (static quantization)

model.export( #format defaults to PyTorch (.pt files), but only TensorRT and openVino format supports uint8 quantization
    format='openvino', #openVINO format is chosen, as TensorRT requires GPU
    dynamic=True, #more compatible for ONNX and TensorRT formats
    batch=8, #model can process 8 images concurrently when inferencing 
    int8=True, #IMPORTANT: activates int8 quantization
    nms=True, #Adds Non-Maximum Suppression(NMS) to the export, prevents duplicating bounding boxes, and +accuracy
    device='cpu',
    data='coco8.yaml' #calibration and validation dataset for quantizing while retaining accuracy 
)