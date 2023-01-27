import cv2
import os
import uuid 

face_cascade = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')

def launch_camera():
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            raise Exception("Encoding failed")
            
    camera.release()
     
    
def capture(filepath):
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    cv2.rectangle(frame, (200, 200), (400, 400), (0, 0, 255), 2)
    filename = '{}.jpg'.format(str(uuid.uuid4()))
    
    if frame is not None and not frame.size == 0:
        cv2.imwrite(os.path.join(os.path.abspath(filepath), filename), frame)
        camera.release()
        return os.path.join(os.path.abspath(filepath), filename)
    else:
        raise Exception("Invalid image received")

def is_human_face(filepath):
    image = cv2.imread(filepath)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    return True if len(faces) else False
