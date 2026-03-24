import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load image
image = cv2.imread('sample_image.jpg')
height, width, _ = image.shape

# Prepare the image for the model
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Perform detection
outs = net.forward(output_layers)

# Analyze detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            print(f'Detected object: {class_id} with confidence: {confidence}')

# Save or display results
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()