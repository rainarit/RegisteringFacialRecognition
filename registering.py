import os
import os.path
import requests
import urllib.request
from requests import get
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.request import urlretrieve
import cv2
import sys
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from matplotlib import pyplot as plt
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import face_recognition
import img
from pygame import mixer
from gtts import gTTS
from selenium.webdriver.support.ui import WebDriverWait
import csv

facePath = '/Users/rraina/Desktop/INTERNSHIP 2019/FacialReg/haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(facePath)
new_image = np.array([])

 def userCreateAndAddToCSV(array, image):
    array.append(image)
    path = "/Users/rraina/Desktop/INTERNSHIP 2019/RegisteringFacialRecognition/"
    os.chdir(path)
    with open("users.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(array)
    return array
    
def newUser():
    url = "https://docs.google.com/forms/d/e/1FAIpQLSdTDTA7wBPT9hFcvbl6Mp-lAF-E3vs1xKJ2P8kwMa5gAbqVAA/viewform"
    driver = webdriver.Chrome(executable_path='/Users/rraina/Desktop/INTERNSHIP 2019/chromedriver')
    newUserMade = False
    while(newUserMade == False):
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(url)
        old_url = driver.current_url
        while True:
            yes_element = driver.find_element_by_xpath("//div[contains(@jscontroller, 'EcW08c')]")
            checked = yes_element.get_attribute("aria-checked") == "true"
            if(checked == True):
                full_name_element = driver.find_element_by_xpath("//input[@aria-label='Full Name:']")
                full_name = full_name_element.get_attribute("data-initial-value")
                company_element = driver.find_element_by_xpath("//input[@aria-label='Company:']")
                company = company_element.get_attribute("data-initial-value")
                ptm_element = driver.find_element_by_xpath("//input[@aria-label='Person to meet:']")
                ptm = ptm_element.get_attribute("data-initial-value")
                driver.find_element_by_xpath("//div[contains(@aria-disabled, 'false')]").click()
                print(full_name)
                print(company)
                print(ptm)
                newUserMade = True
                break;
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    driver.close()
    user_array = [full_name, company, ptm]
    return user_array

def videoing():
    image_came = False
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print("Number of faces:" + str(len(faces)))
        if(len(faces) > 0):
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                img = cv2.rectangle(small_frame, (x, y), (x+w+20, y+h+20), (255, 0, 0), 1)
            # Display the resulting frame
            cv2.imshow('Video', small_frame)
            for n in range(0, len(faces)):
                image1 = img[y:faces[n][1] + faces[n][3], x:faces[n][0] + faces[n][2]]
                new_image = image1
                path = "/Users/rraina/Desktop/INTERNSHIP 2019/RegisteringFacialRecognition/People"
                files = []
                # r=root, d=directories, f = files
                for r, d, f in os.walk(path):
                    for file in f:
                        if '.jpg' in file:
                            files.append(os.path.join(r, file))
                            print("Number of images are:" + str(len(files)))
                if(len(files) > 0):
                    for p in files:
                        image2 = face_recognition.load_image_file(p) 
                        if (len(image1) > 0):
                            if(len(face_recognition.face_encodings(image1)) > 0):
                                encoding_1 = face_recognition.face_encodings(image1)[0]
                                encoding_2 = face_recognition.face_encodings(image2)[0]
                                results_image2 = face_recognition.compare_faces([encoding_1], encoding_2,tolerance=0.40)
                                accuracy_list = face_recognition.face_distance([encoding_1], encoding_2)
                                accuracy = 100 - (accuracy_list[0]*100)
                                header = "Person: " + str(accuracy)
                                print(header)
                                if ((results_image2[0] == True) and (image_came == False)):
                                    print("You're already here man!")
            if(image_came == False or results_image2[0] == False):
                new_user = newUser()
                name = new_user[0] + ".jpg"
                path = "/Users/rraina/Desktop/INTERNSHIP 2019/RegisteringFacialRecognition/People"
                os.chdir(path)
                cv2.imwrite(name, new_image)
                userCreateAndAddToCSV(new_user, new_image)
                video_capture.release()
                cv2.destroyAllWindows()
                break;
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
video_capture = cv2.VideoCapture(0)
videoing()
