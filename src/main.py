import face_recognition
import cv2
import numpy as np


video_capture = cv2.VideoCapture(0)

my_image = face_recognition.load_image_file("./pictures_of_people_i_know/meInUni.jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]


rate_check=False
rate_check_map= list()
false_alarm_check=False
false_alarm_check_map= list()

while True:
    #for rate check using the fmap list
    objToFmap=0
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    font = cv2.FONT_HERSHEY_DUPLEX


    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        objToFmap=1
        matches = face_recognition.compare_faces([my_face_encoding], face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance([my_face_encoding], face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = "Itay Fed"

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    if rate_check == True:
        rate_check_map.append(objToFmap)
    if len(rate_check_map)!=0:
        cv2.putText(frame, "success rate:"+str(sum(rate_check_map)/(len(rate_check_map))), (50, 50), font,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "numer of samples:"+str(len(rate_check_map)), (50, 100), font,
                    1, (255, 0, 0), 2, cv2.LINE_AA)

    if false_alarm_check == True:
        false_alarm_check_map.append(objToFmap)
    if len(false_alarm_check_map)!=0:
        cv2.putText(frame, "number of false alarms:"+str(sum(false_alarm_check_map)), (50, 50), font,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "numer of samples:"+str(len(false_alarm_check_map)), (50, 100), font,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('f'):
        false_alarm_check=not false_alarm_check
    if cv2.waitKey(1) & 0xFF == ord('s'):
        rate_check=not rate_check
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()