from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import threading

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Dummy user database
users = {'123456': 'password123'}

# Load face recognition images
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.write(f'\n{name},{dtString}')

def start_face_recognition():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                markAttendance(name)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    prn = request.form['prn']
    password = request.form['password']
    if prn in users and users[prn] == password:
        session['prn'] = prn
        return redirect(url_for('welcome'))
    return "Invalid PRN or Password"

@app.route('/welcome')
def welcome():
    if 'prn' in session:
        # Start face recognition in a separate thread
        threading.Thread(target=start_face_recognition).start()
        return f"<h1>Welcome, {session['prn']}!</h1>"
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
