#HL_Engine_2
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
import pyowm
import wikipedia
import base64
from xml.dom import minidom

recognizer=cv2.face.LBPHFaceRecognizer_create()
path='dataset'
rate=9600
cap = cv2.VideoCapture(0)

class Advanced_Image_Processing:
    def collectDataSet(filterName,Cam,TargetID):
        faceDetect = cv2.CascadeClassifier(filterName)
        cam = cv2.VideoCapture(Cam)
        id =TargetID
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


    def trainDataSet():
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

        Ids, faces = getImageWithID(path)
        recognizer.train(faces, np.array(Ids))
        recognizer.save('recognizer/trainingdata.yml')
        cv2.destroyAllWindows()


    def lockTarget_IP(filterName,ip,user1,user2,user3,user4,user5):
        faceDetect = cv2.CascadeClassifier(filterName)
        #camera="http://192.168.1.202:8080/video"
        cam = cv2.VideoCapture(ip)
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
                # if(<50):
                if (id == 1):
                    id = user1
                elif (id == 2):
                    id = user2
                elif (id == 3):
                    id = user3
                elif (id == 4):
                    id = user4
                elif (id == 5):
                    id = user5

                else:
                    id = 'unknown'
                # cv2.cv.putText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
                cv2.putText(img, str(id), (x, y + h), font, 2, (255, 0, 0), 3);
            cv2.imshow('face', img)
            if (cv2.waitKey(1) == ord('q')):
                break;
        cam.release()
        cv2.destroyAllWindows()



    def lockTarget_Camera(filterName,camera,user1,user2,user3,user4,user5):
        faceDetect = cv2.CascadeClassifier(filterName)    
        cam = cv2.VideoCapture(camera)
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
                # if(<50):
                if (id == 1):
                    id = user1
                elif (id == 2):
                    id = user2
                elif (id == 3):
                    id = user3
                elif (id == 4):
                    id = user4
                elif (id == 5):
                    id = user5

                else:
                    id = 'unknown'
                # cv2.cv.putText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
                cv2.putText(img, str(id), (x, y + h), font, 2, (255, 0, 0), 3);
            cv2.imshow('face', img)
            if (cv2.waitKey(1) == ord('q')):
                break;
        cam.release()
        cv2.destroyAllWindows()

class Audio_Processing:

    def soundPlayer(location):
        try:
            playsound(location)
        except:
            return ("HLEngine:an issue in playing sound detected")


    def saveAudio(param,location):
        try:
            mytext = param
            language = 'en'
            myobj = gTTS(text=mytext, lang=language, slow=False)
            myobj.save(location)
        except:
            return ("HLEngine:saveAudio issue detected")


    def playAudio(location):
        try:
            pygame.init()
            pygame.mixer.init()
            pygame.mixer.music.load(location)
            pygame.mixer.music.play()
            pygame.event.wait()
        except:
            return ("HLEngine:playAudio issue detected")

    def readText(param):
        try:
            engine = pyttsx3.init()
            engine.getProperty('rate')
            engine.setProperty('rate', 125)
            engine.say(param)
            engine.runAndWait()
        except:
            return ("HLEngine cannot load the required necessay files")


    def readTextSpec(param):
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


            engine.say(param + str(rate))
            engine.runAndWait()
            engine.stop()

        except:

            return("HLEngine: An error occured in readAudioSpec")

class Cipher:
    def dataEncryption(data):
        keyLoader=open('HL_Engine\HL_Crypto\key.txt','r')
        key=keyLoader.read()
        dataModel=[]
        ASCER=(string.printable)
        for i in ASCER:
            dataModel.append(i)
        keyModel=str(key)
        Position_Generator=[]
        Data=str(data)
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


    def dataDecryption(data):
        data=str(data)
        dataModel=[]
        ASCER=(string.printable)
        for i in ASCER:
            dataModel.append(i)
        keyLoader=open('HL_Crypto/key.txt','r')
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
    def find_Port():
        try:
            ser = serial.Serial("COM1", rate)
            print("Connected to COM1")
            return('COM1')
        except:
            print("Disconnected to COM1")
        
        try:
            ser = serial.Serial("COM2", rate)
            print("Connected to COM2")
            return('COM2')
        except:
            print("Disconnected to COM2")

        
        try:
            ser = serial.Serial("COM3", rate)
            print("Connected to COM3")
            return('COM3')
        except:
            print("Disconnected to COM3")


        try:
            ser = serial.Serial("COM4", rate)
            print("Connected to COM4")
            return('COM4')
        except:
            print("Disconnected to COM4")

        try:
            ser = serial.Serial("COM5", rate)
            print("Connected to COM5")
            return('COM5')
        except:
            print("Disconnected to COM5")

        try:
            ser = serial.Serial("COM6", rate)
            print("Connected to COM6")
            return('COM6')
        except:
            print("Disconnected to COM6")


        try:
            ser = serial.Serial("COM7", rate)
            print("Connected to COM7")
            return('COM7')
        except:
            print("Disconnected to COM7")

        try:
            ser = serial.Serial("COM8", rate)
            print("Connected to COM8")
            return('COM8')
        except:
            print("Disconnected to COM8")

        try:
            ser = serial.Serial("COM9", rate)
            print("Connected to COM9")
            return('COM9')
        except:
            print("Disconnected to COM9")

        try:
            ser = serial.Serial("COM10", rate)
            print("Connected to COM10")
            return('COM10')
        except:
            print("Disconnected to COM10")

        try:
            ser = serial.Serial("COM11", rate)
            print("Connected to COM11")
            return('COM11')
        except:
            print("Disconnected to COM11")

        try:
            ser = serial.Serial("COM12", rate)
            print("Connected to COM12")
            return('COM12')
        except:
            print("Disconnected to COM12")

        try:
            ser = serial.Serial("COM13", rate)
            print("Connected to COM13")
            return('COM13')
        except:
            print("Disconnected to COM13")

        try:
            ser = serial.Serial("COM14", rate)
            print("Connected to COM14")
            return('COM14')
        except:
            print("Disconnected to COM14")

        try:
            ser = serial.Serial("COM15", rate)
            print("Connected to COM15")
            return('COM15')
        except:
            print("Disconnected to COM15")

        try:
            ser = serial.Serial("COM16", rate)
            print("Connected to COM16")
            return('COM16')
        except:
            print("Disconnected to COM16")

        try:
            ser = serial.Serial("COM17", rate)
            print("Connected to COM17")
            return('COM17')
        except:
            print("Disconnected to COM17")

        try:
            ser = serial.Serial("COM18", rate)
            print("Connected to COM18")
            return('COM18')
        except:
            print("Disconnected to COM18")

        try:
            ser = serial.Serial("COM19", rate)
            print("Connected to COM19")
            return('COM19')
        except:
            print("Disconnected to COM19")

        try:
            ser = serial.Serial("COM20", rate)
            print("Connected to COM20")
            return('COM20')
        except:
            print("Disconnected to COM20")

        try:
            ser = serial.Serial("/dev/ttyUSB0", rate)
            print("Connected to /dev/ttyUSB0")
            return('/dev/ttyUSB0')
        except:
            print("Disconnected to /dev/ttyUSB0")

        try:
            ser = serial.Serial("/dev/ttyACM0", rate)
            print("Connected to /dev/ttyACM0")
            return('/dev/ttyACM0')
        except:
            print("Disconnected to /dev/ttyACM0")


    def serSend(port,rate,data):
        try:
            ser = serial.Serial(port, rate)
            time.sleep(2)
            ser.write(str.encode(str(data)))        
            return('HLEngine:data sent...')
        except:
            return ("HLEngine:issue with the  port")

    def serRecieve(port,rate):
        try:
            ser = serial.Serial(port, rate)
            Serial_data = ser.readline()
            return (Serial_data)
        except:
            return ("HLEngine:issue with the  port")


    def shutDown_windows():
        try:
            os.system("shutdown /s /t 1")
        except:
            return ("HLEngine :failed to shutdown windows")


    def reboot_windows():
        try:
            os.system("restart /s /t 1")
        except:
            return ("HLEngine :failed to reboot windows")

    def linux_shutdown():
        try:
            os.system("poweroff")
        except:
            return ("HLEngine :failed to shutdown linux")

    def linux_boot():
        try:
            os.system("reboot")
        except:
            return ("HLEngine :failed to reboot linux")

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
    def camSnap(location,frameName,cam):
        try:
            camera = cv2.VideoCapture(cam)
            while True:
                return_value, image = camera.read()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imshow(frameName, gray)
                if cv2.waitKey(1) & 0xFF == ord('x'):
                    cv2.imwrite(location, gray)
                    break
            camera.release()
            cv2.destroyAllWindows()
        except:
            return ("HLEngine:Camera not connected")

    def showImage(location,frameName):
        img = cv2.imread(location)
        cv2.imshow(frameName, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def liveCam_filter(filter,cam,frameName):
        try:
            cap = cv2.VideoCapture(cam)

            # Create the haar cascade
            faceCascade = cv2.CascadeClassifier(filter)
            framer=frameName
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
            return ("HLEngine: An issue with camera or params")

    def videoObjectDetection(cascade,video_source,frameName,objectName):
        try:
            cap = cv2.VideoCapture(video_source)

            # Create the haar cascade
            faceCascade = cv2.CascadeClassifier(cascade)
            framer=frameName
            font = cv2.FONT_HERSHEY_PLAIN
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
                    # flags = cv2.CV_HAAR_SCALE_IMAGEHL_Engine/HLEngine_camSnap.py:40
                )

                #print(format(len(net)))
                # print (len(faces))
                if (len(net) >= 1):
                    #return ("found 2 eyes")
                    cv2.putText(frame, str(objectName), (50, 50), font, 2,
                                (0, 0, 255), 3)



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
            return ("HLEngine: An issue with video_source or params")


    def overlay(dress_png,person_png,finalName_png):
        try:
            img = Image.open(dress_png)

            #print(img.size)

            background = Image.open(person_png)

            #print(background.size)

            # resize the image
            size = (2000,2000)
            background = background.resize(size,Image.ANTIALIAS)

            background.paste(img, (-350, -950), img)
            background.save(finalName_png,"PNG")
            return ('done')

        except:
            return('HLEngine:An issue with the camera or params passed')

class Speech_Recognition_Tool:
    def sR():
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Say something!")
                audio = r.listen(source)
            try:
                print("HLEngine:You said: " + r.recognize_google(audio))
                content = r.recognize_google(audio)
                return(content)

            except sr.UnknownValueError:
                return ("HLEngine:Google Speech Recognition could not understand audio")


            except sr.RequestError as e:
                return ("HLEngine:Could not request results from Google Speech Recognition service; {0}".format(e))
        except:

            return ('HLEngine:microphone not connected')

    def sentiment(param):
        import nltk
        # import TextBlob
        from textblob import TextBlob
        blob1 = TextBlob(param)
        return (blob1.sentiment.polarity)

class Weather_Station:
    def temp(place):
        owm = pyowm.OWM('f8c43bbd601d39c177afabec2d050d04')
        observation = owm.weather_at_place(place)
        weather = observation.get_weather()
        temperature=str(weather.get_temperature('celsius')['temp'])
        return (temperature)

    def sunrise(place):
        owm = pyowm.OWM('f8c43bbd601d39c177afabec2d050d04')
        observation = owm.weather_at_place(place)
        weather = observation.get_weather()
        sunriseTime=str(weather.get_sunrise_time(timeformat='iso'))
        return (sunriseTime)

    def sunset(place):
        owm = pyowm.OWM('f8c43bbd601d39c177afabec2d050d04')
        observation = owm.weather_at_place(place)
        weather = observation.get_weather()
        sunsetTime=weather.get_sunset_time(timeformat='iso')
        return (sunsetTime)

    def humidity(place):
        owm = pyowm.OWM('f8c43bbd601d39c177afabec2d050d04')
        observation = owm.weather_at_place(place)
        weather = observation.get_weather()
        return (weather.get_humidity())

    def wind(place):
        owm = pyowm.OWM('f8c43bbd601d39c177afabec2d050d04')
        observation = owm.weather_at_place(place)
        weather = observation.get_weather()
        return (weather.get_wind())

class WIKI:
    def wiki(param):
        try:
            return(wikipedia.summary(param))
        except:
            return ("HLEngine:error in executing wiki....")

class WordExtractor:
    def FW(param):
        try:
            sent=str(param)
            first, *middle, last = sent.split()
            #print(first, last)
            return(first)
        except:
            return("HLEngine:Error in excuting FW.....")

    def EW(param):
        try:
            sent=str(param)
            first, *middle, last = sent.split()
            #print(first, last)
            return(last)
        except:
            return ("HLEngine:error in executing EW....")

    def Image_decode(location):
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import pytesseract
            import os
            path = location
            img = Image.open(path)
            img = img.convert('RGBA')
            pix = img.load()
            text = pytesseract.image_to_string(Image.open(path))
            #os.remove('temp.jpg')
            return (text)
        except:
            return ("HLEngine:File missing...contact HLadmin")

class XML_Parser:
    def sysArch(question):
        mydoc = minidom.parse('HL_HiveMind/HL_HiveMind_Hub.xml')
        sources = mydoc.getElementsByTagName('source')
        first, *middle, last = question.split()
        for elem in sources:
            x = elem.firstChild.data
            if (x == str(last)):
                checker = int(elem.attributes['name'].value)
                ans = (sources[checker].firstChild.data)
                print("HLEngine:"+ans)
                return(str(ans))
