import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/pi/best0.pt')
video_path = '/home/pi/video/output.avi'
output_dir = '/home/pi/video/cropped_images'
Path(output_dir).mkdir(parents=True, exist_ok=True)
confidence_threshold = 0.5
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
 raise ValueError("Failed to read the video file")
height, width = frame.shape[:2]
output_video_path = '/home/pi/video/detections.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
frame_count = 0
while cap.isOpened():
 ret, frame = cap.read()
 if not ret:
  break
 pil_image = Image.fromarray(frame[..., ::-1])
 results = model([pil_image], size=640)
 pred = results.pred[0]
 if len(pred) == 0:
  continue
 bboxes = pred[:, :4].cpu().numpy()
 class_labels = pred[:, -1].cpu().numpy().astype(int)
 confidences = pred[:, 4].cpu().numpy()
 
 valid_indices = confidences >= confidence_threshold
 bboxes = bboxes[valid_indices]
 class_labels = class_labels[valid_indices]
 
 for bbox, label in zip(bboxes, class_labels):
  xmin, ymin, xmax, ymax = bbox.astype(int)
  cropped_image = pil_image.crop((xmin, ymin, xmax, ymax))
  cropped_image.save(f'{output_dir}/{label}_frame_{frame_count}.jpg')
 
 frame_count += 1
 for bbox, label in zip(bboxes, class_labels):
  xmin, ymin, xmax, ymax = bbox.astype(int)
  cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
  cv2.putText(frame, str(label), (xmin, ymin - 10), 
cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
 output_video.write(frame)
 if frame_count >= 15:
  break
cap.release()
output_video.release()
cv2.destroyAllWindows()
