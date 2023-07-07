import cv2
import numpy as np
from const import CHOOSE_PERCENT, NUMBER_OF_CHOICES, UNANSWERED_CHOICE

def is_checked(cell):
	w, h = cell.shape[0] , cell.shape[1]
	total_pixels = cv2.countNonZero(cell)
	percent = total_pixels / (w * h)
	return percent > CHOOSE_PERCENT

def get_user_answers_by_image(cells):
	question_number = -1
	
	user_answers = [UNANSWERED_CHOICE] * (len(cells) // NUMBER_OF_CHOICES)

	for i, cell in enumerate(cells):
			if i % NUMBER_OF_CHOICES == 0:
				question_number += 1
			if is_checked(cell):
				user_answers[question_number] = i % NUMBER_OF_CHOICES 
	return user_answers

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

def get_boxes_contours(gray_image):

    gray_blur = cv2.GaussianBlur(gray_image, (31, 31), 0) #Noise removal
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

    ## get the inner rectangle (second layers) 
    inner_rectangles = []

    for i, contour in enumerate(contours):
        # Check if the contour has a parent and child contours (second layer)
        if hierarchy[0, i, 3] != -1 and hierarchy[0, i, 2] != -1:
            area = cv2.contourArea(contour) 

            if area > 60000 or area < 44000: # Check the area of contours
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Store the bounding rectangle coordinates
            inner_rectangles.append((x, y, w, h))


    inner_rectangles = sorted(inner_rectangles, key=lambda x: (round(x[0]/300), x[1])) # sort by (x,y) 
    return inner_rectangles
    
def get_boxes(img, number_of_boxes=17):
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inner_rectangles = get_boxes_contours(gray)

    if number_of_boxes > len(inner_rectangles):
        raise ValueError("The number of questions are too much.")

    inner_rectangles = inner_rectangles[ : number_of_boxes]
    boxes = []
    for x, y, w, h in inner_rectangles: ## each rectangle has 10 questions.

        ioa_image = gray.copy()[y: y+h, x:x+w]
        
        thresh = cv2.threshold(ioa_image, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        boxes.extend(split_boxes(thresh))

    return boxes



def get_correct_answers(key_image):
	correct_answers = []
	image = key_image.copy()
	crop_height = 400  # Adjust the value based on your requirements

	# Crop the top portion of the image
	image = image[crop_height:, :, :]

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_green = np.array([40, 40, 40])  # Example lower threshold for green (adjust as needed)
	upper_green = np.array([70, 255, 255])  # Example upper threshold for green (adjust as needed)

    # Create a mask of green regions
	mask = cv2.inRange(hsv, lower_green, upper_green)


	contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])


	for contour in contours:
		x, _, _ , _ = cv2.boundingRect(contour)
		if x < 400:
			correct_answers.append(0)
		elif x < 900:
			correct_answers.append(1)
		elif x < 1200:
			correct_answers.append(2)
		else:
			correct_answers.append(3)
	
	## FIX bug of image. question with number 140 doesn't exist!!!
	correct_answers = correct_answers[:148] + [1] + correct_answers[148:] 
	return correct_answers
