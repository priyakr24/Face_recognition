import cv2
from google.colab.patches import cv2_imshow
from pyagender import PyAgender
from fer import FER

face_cascade =cv2.CascadeClassifier("/content/haarcascade_frontalface_default.xml")
img = cv2.imread("/content/sample.jpg")
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,
                                      minNeighbors=5)



for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)


cv2_imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

agender = PyAgender()
faces = agender.detect_genders_ages(img)
gender =faces[0]['gender']
age =int(faces[0]['age'])
if (gender < 0.5 ):
  print('G_Male')
else:
  print('G_Female')

print('age =',age)


detector = FER()
result = detector.top_emotion(img)
print('Emotion :',result[0])
