import os
import sys
import cv2
import time
from time import sleep
from PIL import Image
from slideshow_final import *
import multiprocessing
import subprocess

e = multiprocessing.Event()
parent_conn, child_conn = multiprocessing.Pipe()
camera_process = None
slideshow_process = None
style_transfer_process = None
elapsed = 0
img_counter = 0


def showTransferVersion(img_name, next_image_path):
	#calls main.py for photo to be processed
	os.system("python3 main.py eval --content-image {} --style-image {} --output-image output/output_{}.jpg --model models/21styles.model --content-size 600 --cuda=0".format(img_name, next_image_path, img_counter+1))




def camPreview(previewName, camID, e):
	img_list = []
	start_time = time.time()
	img_counter = 0
	cv2.namedWindow(previewName)
	cam = cv2.VideoCapture(camID)
	if cam.isOpened():  # try to get the first frame
		rval, frame = cam.read()
	else:
		rval = False

	while rval:
		elapsed = time.time() - start_time
		frame = cv2.resize(frame, (640,480)) #make camera small
		cv2.imshow(previewName, frame)
		rval, frame = cam.read()
		key = cv2.waitKey(20)
		if key == 27:  # exit on ESC
			break
		elif int(elapsed)%secs_per_slide==2:
			img_name = "images/content/opencv_frame_{}.jpg".format(img_counter)
			cv2.imwrite(img_name, frame)
			img_counter += 1

			# created a Pipe() and set slideshow_final as the parent, camera as the child
			# in order to access the next image index from the running slideshow
			next_image_index = parent_conn.recv() + 1 
			next_image_path = load_img_path(images_path)[next_image_index]

			global style_transfer_process
			#run main code on a different process so the cam cv does not halt for 5-6 seconds during processing.
			style_transfer_process = multiprocessing.Process(target=showTransferVersion, args=(img_name,next_image_path,))
			style_transfer_process.start()

			sleep(1) #that's to stop the code from executing too quickly and evaluating the same frame multiple times.
			im = cv2.imread("{}output_{}.jpg".format(images_path, img_counter-1))
			cv2.imshow("Output", im)
	cv2.destroyWindow(previewName)




## Run slideshow and recording on separate processes for both cv windows to be responsive.

def start_recording_proc():
    global camera_process
    camera_process = multiprocessing.Process(target=camPreview, args=("Camera",0,e,))
    camera_process.start()

def slideshow_proc():
	global slideshow_process
	slideshow_process = multiprocessing.Process(target=run_slideshow, args=(child_conn,))
	slideshow_process.start()

start_recording_proc() #Let the show begin
slideshow_proc()







