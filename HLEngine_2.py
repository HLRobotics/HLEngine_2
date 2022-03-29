#HLEngine_2
#Designed and Developed by
#Akhil P Jacob
import os
import cv2
import numpy as np
from PIL import Image
from gtts import gTTS
import pygame
import pyttsx3
from playsound import playsound
import string
import serial
import os
import time
from matplotlib import pyplot as plt
from matplotlib import style
import cv2
import os
import speech_recognition as sr
import wikipedia
from xml.dom import minidom

recognizer=cv2.face.LBPHFaceRecognizer_create()
path='dataset'
rate=9600
cap = cv2.VideoCapture(0)

class Advanced_Image_Processing:

    def __init_(self,filterName,Camera,TargetID,userList):
        self.filterName=filterName
        self.Camera=Camera
        self.TargetID=TargetID
        self.userList=userList
        self.Dataset_path="dataset"

    def collectDataSet(self):
        faceDetect = cv2.CascadeClassifier(self.filterName)
        cam = cv2.VideoCapture(self.Camera)
        id =self.TargetID
        sampleNum = 0
        while (True):
            ret, img = cam.read();
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5);
            for (x, y, w, h) in faces:
                sampleNum += 1
                print(sampleNum)
                cv2.imwrite("dataset/pythface." + str(id) + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.waitKey(100)
            cv2.imshow('Dataset', img)
            cv2.waitKey(1)
            if (sampleNum > 100):
                break
        cam.release()
        cv2.destroyAllWindows()


    def trainDataSet(self):
        def getImageWithID(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faces = []
            IDs = []
            for imagePath in imagePaths:
                faceImg = Image.open(imagePath).convert('L')
                faceNp = np.array(faceImg, 'uint8')
                ID = int(os.path.split(imagePath)[-1].split('.')[1])
                faces.append(faceNp)
                IDs.append(ID)
                cv2.imshow('training', faceNp)
                cv2.waitKey(10)
            return np.array(IDs), faces

        Ids, faces = getImageWithID(self.Dataset_path)
        recognizer.train(faces, np.array(Ids))
        recognizer.save('recognizer/trainingdata.yml')
        cv2.destroyAllWindows()


    def lockTarget_Camera(self):
        faceDetect = cv2.CascadeClassifier(self.filterName)    
        cam = cv2.VideoCapture(self.Camera)
        rec = cv2.face.LBPHFaceRecognizer_create();
        rec.read('recognizer/trainingdata.yml')
        # id=0
        # font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        while (True):
            ret, img = cam.read();
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5);
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, conf = rec.predict(gray[y:y + h, x:x + w])
                for Target in self.userList:
                    ID=self.userList.index(Target)
                    if(ID==id):
                        print(Target,conf)

                # cv2.cv.putText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
                cv2.putText(img, str(id), (x, y + h), font, 2, (255, 0, 0), 3);
            cv2.imshow('HLEngine_2', img)
            if (cv2.waitKey(1) == ord('q')):
                break;
        cam.release()
        cv2.destroyAllWindows()

class Audio_Processing:

    def __init__(self,param,location):
        self.param=param
        self.location=location


    def soundPlayer(self):
        try:
            playsound(self.location)
        except:
            return ("HLEngine_2:an issue in playing sound detected")


    def saveAudio(self):
        try:
            mytext = self.param
            language = 'en'
            myobj = gTTS(text=mytext, lang=language, slow=False)
            myobj.save(self.location)
        except:
            return ("HLEngine_2:saveAudio issue detected")


    def playAudio(self):
        try:
            pygame.init()
            pygame.mixer.init()
            pygame.mixer.music.load(self.location)
            pygame.mixer.music.play()
            pygame.event.wait()
        except:
            return ("HLEngine_2:playAudio issue detected")

    def readText(self):
        try:
            engine = pyttsx3.init()
            engine.getProperty('rate')
            engine.setProperty('rate', 125)
            engine.say(self.param)
            engine.runAndWait()
        except:
            return ("HLEngine_2 cannot load the required necessay files")


    def readTextSpec(self):
        try:


            engine = pyttsx3.init()  # object creation

            """ RATE"""
            rate = engine.getProperty('rate')  # getting details of current speaking rate
            print(rate)  # printing current voice rate
            engine.setProperty('rate', 125)  # setting up new voice rate

            """VOLUME"""
            volume = engine.getProperty('volume')  # getting to know current volume level (min=0 and max=1)
            print(volume)  # printing current volume level
            engine.setProperty('volume', 1.0)  # setting up volume level  between 0 and 1

            """VOICE"""
            voices = engine.getProperty('voices')  # getting details of current voice
            # engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
            engine.setProperty('voice', voices[1].id)  # changing index, changes voices. 1 for female


            engine.say(self.param + str(rate))
            engine.runAndWait()
            engine.stop()

        except:

            return("HLEngine_2: An error occured in readAudioSpec")

class Cipher:

    def __init__(self,enc,dec):
        self.toEncrypt=enc
        self.toDecrypt=dec


    def dataEncryption(self):
        keyLoader=open('HL_Engine\HL_Crypto\key.txt','r')
        key=keyLoader.read()
        dataModel=[]
        ASCER=(string.printable)
        for i in ASCER:
            dataModel.append(i)
        keyModel=str(key)
        Position_Generator=[]
        Data=str(self.toEncrypt)
        for i in Data:
            #print(i)
            if(i in Data):
                Position=dataModel.index(i)
                Position_Generator.append(Position)
        encrypt=[]
        stringer=""
        for i in Position_Generator:        
            data=keyModel[i]
            encrypt.append(data)        
        encrypted=stringer.join(encrypt)    
        return(encrypted)


    def dataDecryption(self):
        data=str(self.toDecrypt)
        dataModel=[]
        ASCER=(string.printable)
        for i in ASCER:
            dataModel.append(i)
        keyLoader=open('HL_Engine\HL_Crypto\key.txt','r')
        key=keyLoader.read()
        keyModel=str(key)
        decrypt=[]
        decryption=[]
        stringer=""
        for i in data:
            if(i in keyModel):
                position=keyModel.index(i)
                decrypt.append(position)
        for i in decrypt:
            decode=dataModel[i]
            decryption.append(decode)

        decrypted=stringer.join(decryption)
        return(decrypted)

class Communications:

    def __init__(self,port,rate,data):
        self.port=port
        self.rate=rate
        self.data=data


    def find_Port(self):
        try:
            ser = serial.Serial("COM1", self.rate)
            print("Connected to COM1")
            return('COM1')
        except:
            print("Disconnected to COM1")
        
        try:
            ser = serial.Serial("COM2", self.rate)
            print("Connected to COM2")
            return('COM2')
        except:
            print("Disconnected to COM2")

        
        try:
            ser = serial.Serial("COM3", self.rate)
            print("Connected to COM3")
            return('COM3')
        except:
            print("Disconnected to COM3")


        try:
            ser = serial.Serial("COM4", self.rate)
            print("Connected to COM4")
            return('COM4')
        except:
            print("Disconnected to COM4")

        try:
            ser = serial.Serial("COM5", self.rate)
            print("Connected to COM5")
            return('COM5')
        except:
            print("Disconnected to COM5")

        try:
            ser = serial.Serial("COM6", self.rate)
            print("Connected to COM6")
            return('COM6')
        except:
            print("Disconnected to COM6")


        try:
            ser = serial.Serial("COM7", self.rate)
            print("Connected to COM7")
            return('COM7')
        except:
            print("Disconnected to COM7")

        try:
            ser = serial.Serial("COM8", self.rate)
            print("Connected to COM8")
            return('COM8')
        except:
            print("Disconnected to COM8")

        try:
            ser = serial.Serial("COM9", self.rate)
            print("Connected to COM9")
            return('COM9')
        except:
            print("Disconnected to COM9")

        try:
            ser = serial.Serial("COM10", self.rate)
            print("Connected to COM10")
            return('COM10')
        except:
            print("Disconnected to COM10")

        try:
            ser = serial.Serial("COM11", self.rate)
            print("Connected to COM11")
            return('COM11')
        except:
            print("Disconnected to COM11")

        try:
            ser = serial.Serial("COM12", self.rate)
            print("Connected to COM12")
            return('COM12')
        except:
            print("Disconnected to COM12")

        try:
            ser = serial.Serial("COM13", self.rate)
            print("Connected to COM13")
            return('COM13')
        except:
            print("Disconnected to COM13")

        try:
            ser = serial.Serial("COM14", self.rate)
            print("Connected to COM14")
            return('COM14')
        except:
            print("Disconnected to COM14")

        try:
            ser = serial.Serial("COM15", self.rate)
            print("Connected to COM15")
            return('COM15')
        except:
            print("Disconnected to COM15")

        try:
            ser = serial.Serial("COM16", self.rate)
            print("Connected to COM16")
            return('COM16')
        except:
            print("Disconnected to COM16")

        try:
            ser = serial.Serial("COM17", self.rate)
            print("Connected to COM17")
            return('COM17')
        except:
            print("Disconnected to COM17")

        try:
            ser = serial.Serial("COM18", self.rate)
            print("Connected to COM18")
            return('COM18')
        except:
            print("Disconnected to COM18")

        try:
            ser = serial.Serial("COM19", self.rate)
            print("Connected to COM19")
            return('COM19')
        except:
            print("Disconnected to COM19")

        try:
            ser = serial.Serial("COM20", self.rate)
            print("Connected to COM20")
            return('COM20')
        except:
            print("Disconnected to COM20")

        try:
            ser = serial.Serial("/dev/ttyUSB0", self.rate)
            print("Connected to /dev/ttyUSB0")
            return('/dev/ttyUSB0')
        except:
            print("Disconnected to /dev/ttyUSB0")

        try:
            ser = serial.Serial("/dev/ttyACM0", self.rate)
            print("Connected to /dev/ttyACM0")
            return('/dev/ttyACM0')
        except:
            print("Disconnected to /dev/ttyACM0")


    def serSend(self):
        try:
            ser = serial.Serial(self.port, self.rate)
            time.sleep(2)
            ser.write(str.encode(str(self.data)))        
            return('HLEngine_2:data sent...')
        except:
            return ("HLEngine_2:issue with the  port")

    def serRecieve(self):
        try:
            ser = serial.Serial(self.port, self.rate)
            Serial_data = ser.readline()
            return (Serial_data)
        except:
            return ("HLEngine_2:issue with the  port")

class System_Control:
    def shutDown_windows():
        try:
            os.system("shutdown /s /t 1")
        except:
            return ("HLEngine_2 :failed to shutdown windows")


    def reboot_windows():
        try:
            os.system("restart /s /t 1")
        except:
            return ("HLEngine_2 :failed to reboot windows")

    def linux_shutdown():
        try:
            os.system("poweroff")
        except:
            return ("HLEngine_2 :failed to shutdown linux")

    def linux_boot():
        try:
            os.system("reboot")
        except:
            return ("HLEngine_2 :failed to reboot linux")
    
class Draw:
    def draw_Line(x,y,name,ycontent,xcontent):
        plt.plot(x,y)#[1,2],[3,4]
        plt.title(name)
        plt.ylabel(ycontent)
        plt.xlabel(xcontent)
        plt.show()

    def draw_Histo(x1,y1,x2,y2,label1,width1,label2,width2,xcontent,ycontent,heading):
        plt.bar(x1,y1,label=label1,width=width1)
        plt.bar(x2,y2,label=label2,width=width2)
        plt.legend()
        plt.xlabel(xcontent)
        plt.ylabel(ycontent)
        plt.title(heading)
        plt.show()

    def draw_scatter(x,y,x1,y1,label1,label2,color1,color2,xlab,ylab,heading): 
        plt.scatter(x,y, label=label1,color=color1)
        plt.scatter(x1,y1,label=label2,color=color2)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.title(heading)
        plt.legend()
        plt.show()

class Image_Processing:

    def __init__(self,location,frameName,camera):
        self.location=location
        self.frameName=frameName
        self.camera=camera

    def camSnap(self):
        try:
            camera = cv2.VideoCapture(self.camera)
            while True:
                return_value, image = camera.read()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imshow(self.frameName, gray)
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    cv2.imwrite(self.location, gray)
                    break
            camera.release()
            cv2.destroyAllWindows()
        except:
            return ("HLEngine_2:Camera not connected")

    def showImage(self):
        img = cv2.imread(self.location)
        cv2.imshow(self.frameName, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def liveCam_filter(self):
        try:
            cap = cv2.VideoCapture(self.camera)

            # Create the haar cascade
            faceCascade = cv2.CascadeClassifier(filter)
            framer=self.frameName
            while (True):

                # Capture frame-by-frame
                ret, frame = cap.read()

                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                net = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                    # flags = cv2.CV_HAAR_SCALE_IMAGE
                )

                #print(format(len(net)))
                # print (len(faces))
                if (len(net) >= 2):
                    return (True)

                # Draw a rectangle around the faces
                for (x, y, w, h) in net:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the resulting frame
                cv2.imshow(framer, frame)
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    break


            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
        except:
            return ("HLEngine_2: An issue with camera or params")

class Image_Overlay:
    def __init__(self,dress_png,person_png,final_png):
        self.dress_png=dress_png
        self.person_png=person_png
        self.final_png=final_png

    def overlay(self):
        try:
            img = Image.open(self.dress_png)

            #print(img.size)

            background = Image.open(self.person_png)

            #print(background.size)

            # resize the image
            size = (2000,2000)
            background = background.resize(size,Image.ANTIALIAS)

            background.paste(img, (-350, -950), img)
            background.save(self.final_png,"PNG")
            return ('done')

        except:
            return('HLEngine_2:An issue with the camera or params passed')    
    
class Speech_Recognition_Tool:
    def sR():
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Say something!")
                audio = r.listen(source)
            try:
                print("HLEngine_2:You said: " + r.recognize_google(audio))
                content = r.recognize_google(audio)
                return(content)

            except sr.UnknownValueError:
                return ("HLEngine_2:Google Speech Recognition could not understand audio")


            except sr.RequestError as e:
                return ("HLEngine_2:Could not request results from Google Speech Recognition service; {0}".format(e))
        except:

            return ('HLEngine_2:microphone not connected | Check pyAudio is Installed (pipwin install PyAudio for windows)')

class Sentimental_Analysis:

    def __init__(self,param):
        self.param=param

    def sentiment(self):
        import nltk
        # import TextBlob
        from textblob import TextBlob
        blob1 = TextBlob(self.param)
        return (blob1.sentiment.polarity)    

class WIKI:

    def __init__(self,word):
        self.word=word

    def wiki(self):
        try:
            return(wikipedia.summary(self.word))
        except:
            return ("HLEngine_2:error in executing wiki....")

class WordExtractor:    
    try:

        def __init__(self,param):
            self.param=param        

        def Extract_Words(self):
            try:
                sent=str(self.param)
                first, middle, last = sent.split()
                #print(first, last)
                return(first,middle,last)
            except:
                return("HLEngine_2:Error in excuting FW.....")

    except:
        print("HLEngine_2: Please Enter the Correct Parameters")
