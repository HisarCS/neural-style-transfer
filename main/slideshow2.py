import cv2
import os

#Image Folder Path
path_folder = "/Users/mertgerdan/desktop/pyRepo/pytorch/pytorch-neural-style/experiments/images/21styles/"

slideshow_width = 600
slideshow_height = 800
#Transition time slideshow
slideshow_trasnition_time = 0.2
#Image stable time
slideshow_img_time = 10
#Window Name
window_name="Famous Images"
#Supoorted formats tuple
supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.dib', '.jpe', '.jp2', '.pgm', '.tiff', '.tif', '.ppm')
#Escape ASCII Keycode
esc_keycode=27
#slide calibration
transit_slides = 15
#minimum weight
min_weight = 0
#maximum weight 
max_weight = 1

#Range function with float 
def range_step(start, step, stop):
	range = start
	while range < stop:
		yield range
		range += step

#Wait Key function with escape handling		
def wait_key(time_seconds):
	#state False if no Esc key is pressed
	state = False
	#Check if any key is pressed. second multiplier for millisecond: 1000
	k = cv2.waitKey(int(time_seconds * 1000))
	#Check if ESC key is pressed. ASCII Keycode of ESC=27
	if k == esc_keycode:  
		#Destroy Window
		cv2.destroyWindow(window_name)
		#state True if Esc key is pressed
		state = True
	#return state	
	return state	
	
#Load image path of all images		
def load_img_path(pathFolder):
	#empty list
	_path_image_list = []
	#Loop for every file in folder path
	for filename in os.listdir(pathFolder):
		#Image Read Path
		_path_image_read = os.path.join(pathFolder, filename)
		#Check if file path has supported image format and then only append to list
		if _path_image_read.lower().endswith(supported_formats):
			_path_image_list.append(_path_image_read)
	#Return image path list
	return _path_image_list

#Load image and return with resize	
def load_img(pathImageRead, resizeWidth, resizeHeight): 	
	#Load an image
	#cv2.IMREAD_COLOR = Default flag for imread. Loads color image.
	#cv2.IMREAD_GRAYSCALE = Loads image as grayscale.
	#cv2.IMREAD_UNCHANGED = Loads image which have alpha channels.
	#cv2.IMREAD_ANYCOLOR = Loads image in any possible format
	#cv2.IMREAD_ANYDEPTH = Loads image in 16-bit/32-bit otherwise converts it to 8-bit
	_img_input = cv2.imread(pathImageRead,cv2.IMREAD_UNCHANGED)
	#Check if image is not empty
	if _img_input is not None:
		#Get read images height and width
		_img_height, _img_width = _img_input.shape[:2]
	
		#if image size is more than resize perform cv2.INTER_AREA interpolation otherwise cv2.INTER_LINEAR for zooming
		if _img_width > resizeWidth or _img_height > resizeHeight:
			interpolation = cv2.INTER_AREA
		else:
			interpolation = cv2.INTER_LINEAR
		
		# perform the actual resizing of the image and show it
		_img_resized = cv2.resize(_img_input, (resizeWidth, resizeHeight), interpolation)
	else:
		#if image is empty
		_img_resized = _img_input
	#return the resized image	
	return _img_resized	


def thread1():
	#Load image paths	
	img_path_list = load_img_path(path_folder)
	#slideshow transition image wait time
	slideshow_transit_wait_time = float (slideshow_trasnition_time) / transit_slides
	#Create a Window
	#cv2.WINDOW_NORMAL = Enables window to resize.
	#cv2.WINDOW_AUTOSIZE = Default flag. Auto resizes window size to fit an image.
	cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
	#Create first image	
	img_one = None	
	#Load every image file path
	for i in range(0,10000):
		#if image is none load image
		if img_one is None:
			#Load first image
			img_one = load_img(img_path_list[i%14], slideshow_width, slideshow_height)
			#Show image in window
			cv2.imshow(window_name, img_one)
			# wait for slide show time to complete and break if Esc key pressed
			if wait_key(slideshow_img_time):
				break
			#continiue to for loop
			continue

		#Load Second image	
		img_two = load_img(img_path_list[i%14], slideshow_width, slideshow_height)
		#for loop through every weight in range 
		for weight_two in range_step(min_weight, float (max_weight)/transit_slides, max_weight):
			#substract weight_two from max_weight to get weight_one 
			weight_one = max_weight - weight_two
			#Weighted addition opertion
			#img_one: First image
			#weight_one: Wight for imge one
			#img_two: Second image
			#weight_two: Wight for imge two
			#0: gamma
			slide_img = cv2.addWeighted(img_one, weight_one, img_two,weight_two, 0)
			#Show image in window
			cv2.imshow(window_name, slide_img)
			# wait for slide show time to complete and clear image path list and also break if Esc key pressed
			if wait_key(slideshow_transit_wait_time):
				del img_path_list[:]
				break
		# wait for slide show time to complete and break if Esc key pressed
		if wait_key(slideshow_img_time):
			break
		#copy image two to image one
		img_one = img_two