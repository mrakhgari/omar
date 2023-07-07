import cv2
import numpy as np
from const import CHOOSE_PERCENT

def is_checked(cell):
	w, h = cell.shape[0] , cell.shape[1]
	total_pixels = cv2.countNonZero(cell)
	percent = total_pixels / (w * h)
	return percent > CHOOSE_PERCENT

def split_boxes(image):
	'''
		Split each box to 10*4 cell. 
	'''
	boxes = []

	h, w = image.shape
	image = image[10 : h-10, 40: w - 10] # remove blank parts

	rows = np.array_split(image, 10, axis=0)
	
	for row in rows:
		cols = np.array_split(row, 4, axis=1)
		for box in cols:
			boxes.append(box)
			# cv2.imshow("rows", box)
			# cv2.waitKey(0)

	return boxes