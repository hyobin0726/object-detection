from __future__ import division
import mediapipe as mp
import cv2
import pygame
from dynamikontrol import Module
import os, sys
from playsound import playsound
import re
import sys
from google.cloud import speech
import pyaudio
from six.moves import queue
global transcript
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path
import pytesseract
from collections import Counter
import removefile
from playsound import playsound
ANGLE_STEP = 1
module = Module()
angle = 0 
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
 min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)
frame_count = 0
start_time = 0
elapsed_time = 0
pygame.mixer.init()
audio_file = "first1.mp3"
pygame.mixer.music.load(audio_file)
audio_played = False 
while cap.isOpened():
 ret, img = cap.read()
 if not ret:
  break
 img = cv2.flip(img, 1) 
 results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 if results.detections:
  frame_count += 1
  if frame_count == 1:
   start_time = cv2.getTickCount()
  for detection in results.detections:
   mp_drawing.draw_detection(img, detection)
   x1 = detection.location_data.relative_bounding_box.xmin
   x2 = x1 + detection.location_data.relative_bounding_box.width
   cx = (x1 + x2) / 2 
   if cx < 0.4: 
    angle += ANGLE_STEP
    module.motor.angle(angle)
   elif cx > 0.6: 
    angle -= ANGLE_STEP
    module.motor.angle(angle)
   if not audio_played:
    pygame.mixer.music.play()
    audio_played = True
   cv2.putText(img, '%d deg' % (angle), org=(10, 30), 
fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=255, thickness=2)
   break
 else:
  frame_count = 0
  start_time = cv2.getTickCount()
 cv2.imshow('Face Cam', img)
 elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
 if elapsed_time > 5 and frame_count > 0:
  cap.release()
  break
 if cv2.waitKey(1) == ord('q'):
  break
pygame.mixer.music.stop()
pygame.mixer.quit()
face_detection.close()
module.disconnect()
cv2.destroyAllWindows()
playsound("second2.mp3")
RATE = 16000
CHUNK = int(RATE / 10)
class MicrophoneStream(object):
 """Opens a recording stream as a generator yielding the audio chunks."""
 def __init__(self, rate, chunk):
  self._rate = rate
  self._chunk = chunk
  self._buff = queue.Queue()
  self.closed = True
 def __enter__(self):
  self._audio_interface = pyaudio.PyAudio()
  self._audio_stream = self._audio_interface.open(
   format=pyaudio.paInt16,
   channels=1,
   rate=self._rate,
   input=True,
   frames_per_buffer=self._chunk,
   stream_callback=self._fill_buffer,
 )
  self.closed = False
  return self
 def __exit__(self, type, value, traceback):
  self._audio_stream.stop_stream()
  self._audio_stream.close()
  self.closed = True
  self._buff.put(None)
  self._audio_interface.terminate()
 def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
  """Continuously collect data from the audio stream, into the buffer."""
  self._buff.put(in_data)
  return None, pyaudio.paContinue
 def generator(self):
  while not self.closed:
   chunk = self._buff.get()
   if chunk is None:
    return
   data = [chunk]
   while True:
    try:
     chunk = self._buff.get(block=False)
     if chunk is None:
      return
     data.append(chunk)
    except queue.Empty:
     break
   yield b"".join(data)
def listen_print_loop(responses):
 """Iterates through server responses and prints them.
 The responses passed is a generator that will block until a response
 is provided by the server.
 Each response may contain multiple results, and each result may contain
 multiple alternatives; for details, see https://goo.gl/tjCPAU. Here we
 print only the transcription for the top alternative of the top result.
 In this case, responses are provided for interim results as well. If the
 response is an interim one, print a line feed at the end of it, to allow
 the next result to overwrite it, until the response is a final one. For the
 final one, print a newline to preserve the finalized transcription.
 """
 global transcript
 num_chars_printed = 0
 for response in responses:
 if response.speech_event_type:
 text = transcript + overwrite_chars
 print('',format(text))
 return text
 if not response.results:
 continue
 result = response.results[0]
 if not result.alternatives:
 continue
 transcript = result.alternatives[0].transcript
 overwrite_chars = " " * (num_chars_printed - len(transcript))
 if not result.is_final:
 sys.stdout.write(transcript + overwrite_chars + "\r")
 sys.stdout.flush()
 num_chars_printed = len(transcript)
 else:
 print(transcript + overwrite_chars)
 if re.search(r"\b(exit|quit)\b", transcript, re.I):
 print("Exiting..")
 break
 num_chars_printed = 0
language_code = "ko-KR" 
global transcript
client = speech.SpeechClient()
config = speech.RecognitionConfig(
 encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
 sample_rate_hertz=RATE,
 language_code=language_code,
)
streaming_config = speech.StreamingRecognitionConfig(
 config=config, single_utterance=True, interim_results=True
)
with MicrophoneStream(RATE, CHUNK) as stream:
 audio_generator = stream.generator()
 requests = (
 speech.StreamingRecognizeRequest(audio_content=content)
 for content in audio_generator
 )
 responses = client.streaming_recognize(streaming_config, requests)
 listen_print_loop(responses)
if(transcript == " "): 유통기한
 playsound("cam.mp3")
 playsound("beep.mp3")
 webcam = cv2.VideoCapture(0) 
 fourcc = cv2.VideoWriter_fourcc(*"XVID")
 output_file = "/home/pi/video/output.avi"
 frame_rate = 30.0
 resolution = (640, 480)
 output = cv2.VideoWriter(output_file, fourcc, frame_rate, resolution)
 capture_time = 5 
 end_time = cv2.getTickCount() + (capture_time * cv2.getTickFrequency())
 while cv2.getTickCount() < end_time:
 ret, frame = webcam.read()
 if ret:
 cv2.imshow("Webcam", frame)
 output.write(frame)
 if cv2.waitKey(1) == ord("q"):
 break
 else:
 break
 webcam.release()
 output.release()
 cv2.destroyAllWindows()
 
 playsound("date.mp3")
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
 
 image_dir = '/home/pi/video/cropped_images'
 pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
 year = []
 month = []
 day = []
 for filename in os.listdir(image_dir):
 if filename.endswith(('.jpg', '.jpeg', '.png')):
 image_path = os.path.join(image_dir, filename)
 image = Image.open(image_path)
 ocr_text = pytesseract.image_to_string(image)
 ocr_text1 = re.sub(r'[a-zA-Z\s.]', '', ocr_text)
 extracted_numbers = re.findall(r'20\d{6}',re.sub(r'[^0-9]', '', ocr_text1))
 print(extracted_numbers)
 for number in extracted_numbers:
 year.append(number[:4])
 month.append(number[4:6])
 day.append(number[6:8])
 most_frequent_year = Counter(year).most_common(1)[0][0]
 most_frequent_month = Counter(month).most_common(1)[0][0]
 most_frequent_day = Counter(day).most_common(1)[0][0]
 final_year = most_frequent_year
 final_month = most_frequent_month
 final_day = most_frequent_day
 from google.cloud import texttospeech
 client = texttospeech.TextToSpeechClient()
 synthesis_input = texttospeech.SynthesisInput(text='{} {} {} 년 월 일입니다
'.format(final_year,final_month,final_day))
 voice = texttospeech.VoiceSelectionParams(
 language_code="ko-KR", name="ko-KR-Wavenet-A"
 )
 audio_config = texttospeech.AudioConfig(
 audio_encoding=texttospeech.AudioEncoding.MP3
 )
 response = client.synthesize_speech(
 input=synthesis_input, voice=voice, audio_config=audio_config
 )
 with open("final.mp3", "wb") as out:
 out.write(response.audio_content)
 print('Audio content written to file "final.mp3"')
 playsound("final.mp3")
 removefile.removefile('/home/pi/video/cropped_images')
