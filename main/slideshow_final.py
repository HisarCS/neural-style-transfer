import cv2
import os

window_width = 800
window_height = 600
window_name = "Famous Images"

secs_per_slide = 10000 #seconds per slide in milliseconds

images_path = "/Users/mertgerdan/desktop/pyRepo/pytorch/pytorch-neural-style/main/images/style-images/"
img_list = []


def load_and_resize_img(name):
	img = cv2.imread("{}{}".format(images_path,name))
	img_resized = cv2.resize(img, (window_width,window_height))
	return img_resized

def load_dir(dir_path, list):
	for file in os.listdir(dir_path):
		try:
			if not file.endswith(".jpg"):
				continue
			else:
				list.append(load_and_resize_img(file))
		except Exception as e:
			print("Error in opening image {}".format(file))

def run_slideshow(conn):
	load_dir(images_path, img_list)
	running = True
	while running:
		#show image and wait x seconds before continuing
		for idx, image in enumerate(img_list):
			conn.send(idx)
			cv2.imshow(window_name, image) 
			key = cv2.waitKey(secs_per_slide)
			if key == 27: #ESC
				running = False
				break

def load_img_path(pathFolder):
	#empty list
	_path_image_list = []
	#Loop for every file in folder path
	for filename in os.listdir(pathFolder):
		#Image Read Path
		_path_image_read = os.path.join(pathFolder, filename)
		#Check if file path has supported image format and then only append to list
		if not _path_image_read.lower().endswith(".jpg"):
			continue
		else:
			_path_image_list.append(_path_image_read)
	#Return image path list
	return _path_image_list