import numpy as np
import cv2
import face_recognition as fr

video_capture = cv2.VideoCapture(0)



messi_image= fr.load_image_file("messi.jpg")
messi_face_encoding = fr.face_encodings(messi_image)[0]



known_face_encoding = [rushabh_face_encoding,messi_face_encoding]
known_face_names = ["Rushabh","Messi"]


while True:
    check,frames = video_capture.read()

    face_locations = fr.face_locations(frames)
    face_encodings = fr.face_encodings(frames,face_locations)

    for (top,right,bottom,left),face_encodings in zip(face_locations,face_encodings):

        matches = fr.compare_faces(known_face_encoding,face_encodings)

        name = "Unknown"

        face_distance = fr.face_distance(known_face_encoding,face_encodings)

        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(frames,(left,top),(right,bottom),(0,0,255),3)

        cv2.rectangle(frames,(left,bottom -35),(right,bottom),(0,0,255),cv2.FILLED)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frames,name,(left,bottom - 6),font,1.0,(255,255,255),1)

    resized = cv2.resize(frames,(500,500))
    cv2.imshow("Face Recognition",resized)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()