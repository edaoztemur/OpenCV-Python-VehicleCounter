# -*- coding: utf-8 -*-
"""
Created on Wed May 25 22:16:44 2022

@author: edaoz
"""

import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")

#Arka planı silinmiş Kare oluşturur.
fgbg = cv2.createBackgroundSubtractorMOG2()


kernel = np.ones((5,5),np.uint8)

class Coordinate: #Kordinatları saklamak için class oluşturduk.
    def __init__(self,x,y):
        self.x=x
        self.y=y

class Sensor:
    def __init__(self,Coordinate1,Coordinate2,Square_width,Square_lenght):
        self.Coordinate1 = Coordinate1
        self.Coordinate2 = Coordinate2
        self.Square_width = Square_width #karenin genişliği
        self.Square_lenght = Square_lenght #karenin uzunluğu
        #maskenin alanı 
        self.Mask_Domain = abs(self.Coordinate2.x-Coordinate1.x)*abs(self.Coordinate2.y-self.Coordinate1.y) #abs negatifleri önlemek için kullanıldı.
        self.Mask = np.zeros((Square_lenght,Square_width,1),np.uint8)
        cv2.rectangle(self.Mask, (self.Coordinate1.x,self.Coordinate1.y),(self.Coordinate2.x,self.Coordinate2.y),(255),thickness=cv2.FILLED)
        self.situation = False #durum değerlendirmesi
        self.Car_Counter = -1 #algılanan araç sayısı


Sensor1 = Sensor(Coordinate(300,180),Coordinate(450,240),1080,250) #sensor1 koordinatları
Sensor2 = Sensor(Coordinate(100,180),Coordinate(250,240),1080,250) #sensor2 koordinatları

font = cv2.FONT_HERSHEY_DUPLEX #yazı fontu

      

while (1):
    ret, square = cap.read() #araçları tanıyacağımız kare
    
    #kesilmiş video boyutu
    cropped_video = square[350:600,100:1180]  #Video yu kestik daha öncesinde shape özelliğiyle kordinatlara eriştik.

 
    Frame1 = fgbg.apply(cropped_video) #arka planı silinmiş kare
    Frame1 = cv2.morphologyEx(Frame1, cv2.MORPH_OPEN, kernel) #Arka planı silinmiş kareye opening uygulandı.
    ret, Frame1 = cv2.threshold(Frame1, 80, 255, cv2.THRESH_BINARY)
    
    #konturleri bulmak için
    cnts,_ = cv2.findContours(Frame1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    #sonucu yazdırdığımız ekran 
    Conclusion = cropped_video.copy() #Sonuctaki degerleri kopyalar.

    Doldurulmus_Resim = np.zeros((cropped_video.shape[0],cropped_video.shape[1],1),np.uint8) # Siyahlarla doldurduğumuz kare

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if(w>30 and h>30): #boyut sınırı için
            cv2.rectangle(Conclusion,(x,y),(x+w,y+h),(0,255,0),thickness=4) #arabaları dikdörtgen içine aldık
            cv2.rectangle(Doldurulmus_Resim,(x,y),(x+w,y+h),(255),thickness=cv2.FILLED)

    #Sensor 1 sol şaritteki araçları sayıyor.
    Sensor1_Mask_Conclusion = cv2.bitwise_and(Doldurulmus_Resim,Doldurulmus_Resim,mask=Sensor1.Mask)
    #beyaz piksel sayısı [frame içinde]
    Sensor1_white_pixel_count = np.sum(Sensor1_Mask_Conclusion==255) #Beyaz piksellerin sayisini bulur.
    #Bulunan beyaz piksel sayisini Masknin boyutuna bölüp oran çıkarır. Bu sayede Sensor üzerinden araç geçtikce oran artar.
    Sensor1_ratio = Sensor1_white_pixel_count/Sensor1.Mask_Domain  
   
#sensör 1 için verilen oranlar
    if(Sensor1_ratio>=0.75 and Sensor1.situation==False): # araç geçmezken kırmızı kutu
        cv2.rectangle(Conclusion, (Sensor1.Coordinate1.x, Sensor1.Coordinate1.y), (Sensor1.Coordinate2.x, Sensor1.Coordinate2.y),
                      (0, 255, 0), thickness=cv2.FILLED)
        Sensor1.situation=True
    elif(Sensor1_ratio<=0.75 and Sensor1.situation==True):  # araç geçtiğinde yeşil kutu yanar.
        cv2.rectangle(Conclusion, (Sensor1.Coordinate1.x, Sensor1.Coordinate1.y), (Sensor1.Coordinate2.x, Sensor1.Coordinate2.y),
                      (0, 0, 255), thickness=cv2.FILLED)
        Sensor1.situation=False #sensor 1 için durum false ise
        Sensor1.Car_Counter+=1 #Sensor situation false iken yani üzerinden araç geçtiğinde araç sayisini artir.
    else:
        cv2.rectangle(Conclusion, (Sensor1.Coordinate1.x, Sensor1.Coordinate1.y), (Sensor1.Coordinate2.x, Sensor1.Coordinate2.y),
                      (0, 0, 255), thickness=cv2.FILLED)
    cv2.putText(Conclusion,str(Sensor1.Car_Counter),(Sensor1.Coordinate1.x,Sensor1.Coordinate1.y+60),font,3,(255,255,255)) #Algılanan araç Sayisini yazidir.
    
    #Sensor 2 sağ şaritteki araçları sayıyor.
    Sensor2_Mask_Conclusion = cv2.bitwise_and(Doldurulmus_Resim,Doldurulmus_Resim,mask=Sensor2.Mask)
    #beyaz piksel sayısı [frame içinde]
    Sensor2_white_pixel_count = np.sum(Sensor2_Mask_Conclusion==255) # Beyaz piksellerin sayisini bulur.
    #Bulunan beyaz piksel sayisini Masknin boyutuna bölüp oran çıkarır. Bu sayede Sensor üzerinden araç geçtikce oran artar.
    Sensor2_Oran = Sensor2_white_pixel_count/Sensor2.Mask_Domain
   
    if(Sensor2_Oran>=0.75 and Sensor2.situation==False): # araç geçmezken kırmızı kutu
        cv2.rectangle(Conclusion, (Sensor2.Coordinate1.x, Sensor2.Coordinate1.y), (Sensor2.Coordinate2.x, Sensor2.Coordinate2.y),
                      (0, 255, 0), thickness=cv2.FILLED)
        Sensor2.situation=True
    elif(Sensor2_Oran<=0.75 and Sensor2.situation==True):  # araç geçtiğinde yeşil kutu yanar.
        cv2.rectangle(Conclusion, (Sensor2.Coordinate2.x, Sensor2.Coordinate1.y), (Sensor2.Coordinate2.x, Sensor2.Coordinate2.y),
                      (0, 0, 255), thickness=cv2.FILLED)
        Sensor2.situation=False #sensor 2 için durum false ise
        Sensor2.Car_Counter+=1 #Sensor situation false iken yani üzerinden araç geçtiğinde araç sayisini artir.
    else:
        cv2.rectangle(Conclusion, (Sensor2.Coordinate1.x, Sensor2.Coordinate1.y), (Sensor2.Coordinate2.x, Sensor2.Coordinate2.y),
                      (0, 0, 255), thickness=cv2.FILLED)
    cv2.rectangle(Conclusion, (Sensor2.Coordinate1.x+450, Sensor2.Coordinate1.y-70), (Sensor2.Coordinate2.x+750, Sensor2.Coordinate2.y-90),
                  (193,255,193), thickness=cv2.FILLED)
    cv2.rectangle(Conclusion, (Sensor2.Coordinate1.x+567, Sensor2.Coordinate1.y-20), (Sensor2.Coordinate2.x+570, Sensor2.Coordinate2.y),
                  (131,139,139), thickness=cv2.FILLED)

    cv2.putText(Conclusion,str(Sensor2.Car_Counter),(Sensor2.Coordinate1.x,Sensor2.Coordinate1.y+60),font,3,(255,255,255)) #Algılanan araç Sayisini yazidir.
    cv2.putText(Conclusion,'TOPLAM GECEN ARAC',(Sensor2.Coordinate1.x+500,Sensor2.Coordinate1.y-40),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255))#sağ şeritteki araç sayisi
    cv2.putText(Conclusion,str(Sensor2.Car_Counter+Sensor1.Car_Counter),(Sensor2.Coordinate1.x+570,Sensor2.Coordinate1.y+60),font,3,(255,255,255)) # toplam arac sayisi
    
   #cv2.imshow("Kare",square)
    cv2.imshow("Kesilmiş Kare",cropped_video)
    cv2.imshow("Frame", Frame1)
    cv2.imshow("Doldurulmuş Resim",Doldurulmus_Resim)
    cv2.imshow("Conclusion",Conclusion) #sensor1 ve sensor2 için sınuç yazdırdığımız ekran
    
        ######################
#esc tusuna basılınca cıkmak için eklendi yoksa ekran gri ekranla kilitleniyor.
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
      #######################
      
      
cap.release()
cv2.destroyAllWindows()

