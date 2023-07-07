import cv2
import numpy as np
from utils import split_boxes, is_checked
import os
import const


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
    # cv2.imshow('b', boxes[1])
    # cv2.waitKey(0)
    # cv2.imshow('b', boxes[-1])
    # cv2.waitKey(0)

    return boxes

def score(user_answers: list, correct_answers: list) -> float:
    question_number = min(len(user_answers), len(correct_answers)) ## Check the same size
    user_answers = user_answers[:question_number]
    correct_answers = correct_answers[:question_number]

    tp = sum(1 for x, y in zip(user_answers, correct_answers) if x == y)

    return tp / question_number * 100


def representation(user_answers, correct_answers, test_image):
    question_number = min(len(user_answers), len(correct_answers))

    image = test_image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inner_rectangles = get_boxes_contours(gray)

    for qn in range(question_number):
        box_number = qn // const.NUMBER_OF_QUESTION_IN_BOX
        row_number = qn % const.NUMBER_OF_QUESTION_IN_BOX

        x, y, _, _ = inner_rectangles[box_number]
        
        color = (0, 0, 255) if user_answers[qn] != correct_answers[qn] else (0, 255, 0) # red for wrong answers and green for correct
        t_x, t_y = x + 50 + 40 * correct_answers[qn] , y + 20 + 29 * row_number 

        cv2.ellipse(image, (t_x, t_y), (17,10), 0, 0 , 360, color, -1)

    image = cv2.resize(image, (800, 800))
    cv2.imshow("image", image)
    cv2.waitKey(0)

def is_tick_pattern(contour):
    # Calculate the contour perimeter
    perimeter = cv2.arcLength(contour, True)

    # Approximate the contour shape with a polygon
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the contour has 3 line segments and an angle close to 90 degrees
    if len(approx) == 3:
        angles = []
        for i in range(3):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % 3][0]
            p3 = approx[(i + 2) % 3][0]
            v1 = p1 - p2
            v2 = p3 - p2
            dot_product = v1.dot(v2)
            angle = abs(np.arccos(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi)
            angles.append(angle)
        avg_angle = sum(angles) / len(angles)
        if abs(avg_angle - 90) < 10:
            return True
    
    return False

def get_correct_answers(key_image):
    correct_answers = []
    image = key_image.copy()
    crop_height = 400  # Adjust the value based on your requirements

    # Crop the top portion of the image
    image = image[crop_height:, :, :]
    key_image = key_image[crop_height:, :, :]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])  # Example lower threshold for green (adjust as needed)
    upper_green = np.array([70, 255, 255])  # Example upper threshold for green (adjust as needed)

    # Create a mask of green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)


    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(key_image, contours, -1 , (255, 0, 0), 5)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])


    for contour in contours:
        x, _, _ , _ = cv2.boundingRect(contour)
        print(x)
        if x < 400:
            correct_answers.append(0)
        elif x < 900:
            correct_answers.append(1)
        elif x < 1200:
            correct_answers.append(2)
        else:
            correct_answers.append(3)
    return correct_answers



key_image = cv2.imread('data/ResponseLetter/kild.png')
correct_answers = get_correct_answers(key_image)
# Fix bug in image (Question 149 doesn't exists!)
correct_answers = correct_answers[:148] + [1] + correct_answers[148:]

for file in os.listdir('data/ResponseLetter/'):
    if 'kild' in file:
        continue

    image = cv2.imread('data/ResponseLetter/'+file)
    
    cells = get_boxes(image)

    user_answers = [const.UNANSWERED_CHOICE] * (len(cells) // const.NUMBER_OF_CHOICES)
    question_number = -1
    for i, cell in enumerate(cells):
        if i % const.NUMBER_OF_CHOICES == 0:
            question_number += 1
        if is_checked(cell):
            user_answers[question_number] = i % const.NUMBER_OF_CHOICES 
        

    user_score = score(user_answers, correct_answers)
    # represe
    # ntation(user_answers, correct_answers, image)
    # break