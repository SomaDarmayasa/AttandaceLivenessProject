import cv2
import os
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from datetime import datetime
from scipy.spatial import distance as dist
import dlib
import imutils
from imutils import face_utils


def selesai1():
    intructions.config(text="Rekam Data Telah Selesai!")


def selesai2():
    intructions.config(text="Training Wajah Telah Selesai!")


def selesai3():
    intructions.config(text="Absensi Telah Dilakukan")


def rekamDataWajah():
    wajahDir = 'datasetwajah'
    wajahDirTest = 'datatestwajah'

    # membuka camera
    cam = cv2.VideoCapture(0)
    cam.set(3, 648)  # ubah lebar video
    cam.set(4, 348)  # ubah tinggi video
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faceID = entry2.get()
    nama = entry1.get()
    nim = entry2.get()
    kelas = entry3.get()

    ambilData = 0  # membuat variable yang menampung jumlah data bg gray

    # merekam dataset gambar format bg grayscale
    while True:  # selama true akan melooping dan camera akan menyala
        retV, frame = cam.read()  # mengambil frame dari camera dan ditampilkan
        # variabel untuk membuat gambar bg menjadi gray
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(
            abuabu, 1.3, 5)  # frame ,scalefactor, minNeighboar

        for (x, y, w, h) in faces:  # membuat perulangan dengan parameter x,y titik temu dan w = width(lebar) dan h= height(tinggi) dari gambar
            frame = cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # membuat kotak yang akan mendetect wajah
            ambilData += 1  # mengambil 1 data gambar tiap frame
            namaFile = str(nim) + '_'+str(nama) + '_' + \
                str(kelas) + '_' + str(ambilData) + '.jpg'
            # menyimpan data wajah ke dalam folder dataset
            cv2.imwrite(wajahDir + '/' + namaFile, abuabu[y:y+h, x:x+w])

        # menampilkan camera serta gambar yg dihasilkan kamera
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
            break
        elif ambilData > 70:  # jika data sudah diambil sebanyak 70 maka akan perekaman akan berhenti
            break

    # membuat variable yang menampung jumlah data bg color
    ambilData = 0
    while True:
        retV, frame = cam.read()
        abuabu = frame
        faces = faceDetector.detectMultiScale(abuabu, 1.3, 5)

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ambilData += 1
            namaFile = str(nim) + '_'+str(nama) + '_' + \
                str(kelas) + '_' + str(ambilData) + '.jpg'
            # menyimpan data wajah ke dalam folder datatest
            cv2.imwrite(wajahDirTest + '/' + namaFile, abuabu[y:y+h, x:x+w])

        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
            break
        elif ambilData > 30:  # mengambil 30 dataset format RGB
            break
    selesai1()
    cam.release()
    cv2.destroyAllWindows()  # untuk menghapus data yang sudah dibaca


def trainingWajah():
    # menentukan path folder untuk foto yang sudah diambil di dalam folder database
    wajahDir = 'datasetwajah'
    latihDir = 'train'

    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # fungsi untuk mengambil image serta mengconvert ke array

    def getImageLabel(path):
        # library os untuk membaca file dari path dataset
        # menjoinkan file dataset ke dalam parameter path yang ada pada fungsi getImages...
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []  # data sample akan disimpan ke dalam list array kosong
        faceIDs = []  # data faceId akan disimpan ke dalam list array kosong

        # mengambil seluruh file  yang telah dibaca kedalam parameter path yang ada di fungsi getImageLabel
        for imagePath in imagePaths:
            # library PILimage untuk membuka file dalam path dan mengconvert ke dalam grayscale
            PILimg = Image.open(imagePath).convert('L')
            # menyimpan variable PIL_img ke dalam numpy array
            imgNum = np.array(PILimg, 'uint8')

            # format file dipisah -1(dari kanan), kita split hilangkan extention .jpg dengan "."
            faceID = int(os.path.split(imagePath)[-1].split('_')[0])

            # mengambil variable img_numpy setelah itu detect menggunakan variabel detector diatas
            faces = faceDetector.detectMultiScale(imgNum)

            for (x, y, w, h) in faces:  # looping variable faces dengan parameter x,y,w,h

                # menambahkan img_numpy yg sudah di detect ke dalam facesamples
                faceSamples.append(imgNum[y:y + h, x:x + w])
                # menambahkan id yang sudah di split atau menghilangkan extention jpg ke dalam variable ids
                faceIDs.append(faceID)

            return faceSamples, faceIDs  # mengembalikan facesamples dan id

    # proses training
    # faces dan ids akan disimpan ke dalam path dataset
    faces, IDs = getImageLabel(wajahDir)
    # mentraining faces dan ids(dlm bntuk numpy array) dengan variable Facerecognizer
    faceRecognizer.train(faces, np.array(IDs))
    # simpan
    # menyimpan model hasil training ke dalam folder train/trainer.yml
    faceRecognizer.write(latihDir + '/trainer70.yml')
    selesai2()


def markAttendance(name):
    with open("Attendance.csv", 'r+') as f:
        namesDatalist = f.readlines()
        namelist = []
        yournim = entry2.get()
        yourclass = entry3.get()
        for line in namesDatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{yourclass},{yournim},{dtString}')


def FaceRecognition():

    latihDir = 'train'
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read(latihDir + '/trainer70.yml')
    font = cv2.FONT_HERSHEY_SIMPLEX

    """ 
    dlib describes eye with 6 points.
    when you blink, the EAR value will change from 0.3 to  near 0.05
    """

    # if you have glasses, you should cahnge the threshold!
    Eye_AR_Thresh = 0.2

    # how many frames shows the blink( use it for reduce noises)
    # berapa frame yang ditampilkan pada saat berkedip
    Eye_AR_Consec_frames = 3

    # initialize the frame counters and the total number of blinks
    # inisialisasi counter frame untuk menambah jumlah kedipan
    counter = 0
    total = 0
    id = " "

    def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of
        # menghitung jarak euclidean distance diantara dua himpunan
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A+B) / (2*C)

        # return the eye aspect ratio
        return ear
        # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    #id = 0
    yourname = entry1.get()
    names = []
    names.append(yourname)

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    weightMin = 0.1*cam.get(3)  # weightmin
    heightMin = 0.1*cam.get(4)  # heightmin
    while True:
        retV, frame = cam.read()
        frame = cv2.flip(frame, 1)
        if frame is None:
            break
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        frame = imutils.resize(frame, width=500)
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(
            abuabu, 1.2, 5, minSize=(round(weightMin), round(heightMin)), )
        # detect faces in the grayscale frame
        rects = detector(abuabu, 0)
        for (x, y, w, h) in faces:
            # loop over the face detections
            for face in rects:
                (x1, y1) = (face.left(), face.top())
                (x2, y2) = (face.right(), face.bottom())

                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(abuabu, face)
                shape = face_utils.shape_to_np(shape)

                # find left and right eyes and calculate the EAR
                # left eye is 37-42th points (numpy starts from 0)
                leftEye = shape[36:42]
                rightEye = shape[42:48]

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                EAR = (leftEAR + rightEAR) / 2

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if EAR < Eye_AR_Thresh:
                    counter += 1
                    id, confidance = faceRecognizer.predict(
                        abuabu[y:y+h, x:x+w])
                    """menggunakan variabel id berdasarkan id yg 
                        direkam dan confidance untuk mempredict dari file train"""

                    if (confidance < 60):  # jika nilai confidence kurang dari 69 ,bagusnya <60
                        # maka digunakan variable nama sesuai id yang telah dibuat
                        id = names[id]
                    else:  # selain itu
                        # maka digunakan variable nama dengan index pertama adalah 0
                        id = names[0]
                    # id, confidence = faceRecognizer.predict(
                    #     abuabu[y:y+h, x:x+w])
                    # if (confidence < 100):
                    #     id = names[0]
                    #     confidence = "  {0}%".format(round(150 - confidence))
                    # elif confidence < 50:
                    #     id = names[0]
                    #     confidence = "  {0}%".format(round(170 - confidence))

                    # elif confidence > 70:
                    #     id = "Tidak Diketahui"
                    #     confidence = "  {0}%".format(round(150 - confidence))

                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if counter > Eye_AR_Consec_frames:
                        total += 1

                    # reset the eye frame counter
                    counter = 0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # draw the total number of blinks on the frame along with
                # the computed eye aspect ratio for the frame
                cv2.putText(frame, "Blinks: {}".format(total),
                            (10, 20), font, 0.55, (0, 0, 255), 1)
                cv2.putText(frame, "EAR: {:.2f}".format(EAR),
                            (10, 50), font, 0.55, (0, 0, 255), 1)
                cv2.putText(frame, str(id), (x + 5, y - 5),
                            font, 1, (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # cv2.putText(frame, str(confidence), (x + 5, y + h + 25),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('ABSENSI WAJAH', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
            break
    markAttendance(id)
    selesai3()
    cam.release()
    cv2.destroyAllWindows()


# GUI
root = tk.Tk()
# mengatur canvas (window tkinter)
canvas = tk.Canvas(root, width=700, height=400)
canvas.grid(columnspan=3, rowspan=8)
canvas.configure(bg="black")
# judul
judul = tk.Label(root, text="Face Attendance - Smart Absensi",
                 font=("Roboto", 34), bg="#242526", fg="white")
canvas.create_window(350, 80, window=judul)

# credit
# made = tk.Label(root, text="Made by Alvin Aprianto", font=(
#     "Times New Roman", 13), bg="black", fg="white")
# canvas.create_window(360, 20, window=made)

# for entry data nama
entry1 = tk.Entry(root, font="Roboto")
canvas.create_window(457, 170, height=25, width=411, window=entry1)
label1 = tk.Label(root, text="Nama Siswa",
                  font="Roboto", fg="white", bg="black")
canvas.create_window(90, 170, window=label1)
# for entry data nim
entry2 = tk.Entry(root, font="Roboto")
canvas.create_window(457, 210, height=25, width=411, window=entry2)
label2 = tk.Label(root, text="NIM", font="Roboto", fg="white", bg="black")
canvas.create_window(60, 210, window=label2)
# for entry data kelas
entry3 = tk.Entry(root, font="Roboto")
canvas.create_window(457, 250, height=25, width=411, window=entry3)
label3 = tk.Label(root, text="Kelas", font="Roboto", fg="white", bg="black")
canvas.create_window(65, 250, window=label3)

global intructions

# tombol untuk rekam data wajah
intructions = tk.Label(root, text="Welcome", font=(
    "Roboto", 15), fg="white", bg="black")
canvas.create_window(370, 300, window=intructions)
Rekam_text = tk.StringVar()
Rekam_btn = tk.Button(root, textvariable=Rekam_text, font="Roboto",
                      bg="#20bebe", fg="white", height=1, width=15, command=rekamDataWajah)
Rekam_text.set("Take Images")
Rekam_btn.grid(column=0, row=7)

# tombol untuk training wajah
Rekam_text1 = tk.StringVar()
Rekam_btn1 = tk.Button(root, textvariable=Rekam_text1, font="Roboto",
                       bg="#20bebe", fg="white", height=1, width=15, command=trainingWajah)
Rekam_text1.set("Training")
Rekam_btn1.grid(column=1, row=7)

# tombol absensi dengan wajah
Rekam_text2 = tk.StringVar()
Rekam_btn2 = tk.Button(root, textvariable=Rekam_text2, font="Roboto",
                       bg="#20bebe", fg="white", height=1, width=20, command=FaceRecognition)
Rekam_text2.set("Automatic Attendance")
Rekam_btn2.grid(column=2, row=7)

root.mainloop()
