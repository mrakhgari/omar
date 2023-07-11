import cv2
import numpy as np
from utils import split_boxes, is_checked, get_correct_answers, get_boxes_contours, get_boxes, get_user_answers_by_image
import os
import const
import csv
import imghdr

class ClassScore:
    def __init__(self, test_path, key_path) -> None:
        self.__test_path = test_path
        self.__key_path = key_path
        self.test_image = cv2.imread(self.__test_path)
        self.key_image = cv2.imread(self.__key_path)
    
    
    def score(self, user_answers: list, correct_answers: list) -> float:
        question_number = min(len(user_answers), len(correct_answers)) ## Check the same size
        user_answers = user_answers[:question_number]
        correct_answers = correct_answers[:question_number]

        tp = sum(1 for x, y in zip(user_answers, correct_answers) if x == y)

        return tp / question_number * 100

    
    def representation(self, test_image, key_image):
        cells = get_boxes(test_image)
        correct_answers = get_correct_answers(key_image)

    
        user_answers = get_user_answers_by_image(cells)
        question_number = min(len(user_answers), len(correct_answers))

        image = test_image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inner_rectangles = get_boxes_contours(gray)

        for qn in range(question_number):
            box_number = qn // const.NUMBER_OF_QUESTION_IN_BOX
            row_number = qn % const.NUMBER_OF_QUESTION_IN_BOX

            x, y, _, _ = inner_rectangles[box_number]
            
            color = (0, 0, 255) if user_answers[qn] != correct_answers[qn] else (0, 255, 0) # red for wrong answers and green for correct
            t_x, t_y = x + 50 + 38 * correct_answers[qn] , y + 20 + 29 * row_number 

            cv2.ellipse(image, (t_x, t_y), (17,10), 0, 0 , 360, color, -1)

        image = cv2.resize(image, (800, 800))
        cv2.imshow("result_image", image)
        cv2.waitKey(0)

    def save_status(self):
        cells = get_boxes(self.test_image)
        user_answers = get_user_answers_by_image(cells)
        correct_answers = get_correct_answers(self.key_image)

        with open(self.__test_path+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i, answer in enumerate(correct_answers):
                if user_answers[i] == const.UNANSWERED_CHOICE:
                    data = [i+1, '-']
                elif user_answers[i] == answer:
                    data = [i+1, 'True']
                else:
                    data = [i+1, 'False']

                writer.writerow(data)

    @staticmethod
    def save_all_status(test_dir_path, key_path):
        key_image = cv2.imread(key_path)

        correct_answers = get_correct_answers(key_image)
        
        with open('all_status.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image_name'] + list(range(1, len(correct_answers)+1 )))

            for file in os.listdir(test_dir_path):
                
                if not imghdr.what(f'{test_dir_path}/{file}'): ## is not an image
                    continue

                if 'kild' in file: 
                    continue ## Pass the key
                image = cv2.imread(f'{test_dir_path}/{file}')
                cells = get_boxes(image)
                user_answers = get_user_answers_by_image(cells)
                data = [file]
                for i, answer in enumerate(correct_answers):
                    if user_answers[i] == const.UNANSWERED_CHOICE:
                        data.append('-')
                    elif user_answers[i] == answer:
                        data.append('True')
                    else:
                        data.append('False')

                writer.writerow(data)

    @staticmethod
    def save_all(test_dir_path, key_path):
        key_image = cv2.imread(key_path)

        correct_answers = get_correct_answers(key_image)
        with open('all_score.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['name', 'score'])

            for file in os.listdir(test_dir_path):
                if not imghdr.what(f'{test_dir_path}/{file}'): ## is not an image
                    continue
                if 'kild' in file: 
                    continue ## Pass the key
                _instance = ClassScore(f'{test_dir_path}/{file}', key_path)
                cells = get_boxes(_instance.test_image)
                user_answers = get_user_answers_by_image(cells)
                
                score = _instance.score(user_answers, correct_answers)
                writer.writerow([file, score])


if __name__ == '__main__':
    test = ClassScore('data\ResponseLetter\image0000050A.tif', 'data\ResponseLetter\kild.png')
    test.representation(test.test_image, test.key_image)
    test.save_status()

    # ClassScore.save_all_status('data\ResponseLetter\\', 'data\ResponseLetter\kild.png')
    # ClassScore.save_all('data\ResponseLetter\\', 'data\ResponseLetter\kild.png')
