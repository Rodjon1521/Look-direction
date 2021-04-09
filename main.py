import cv2
import numpy as np
import dlib

font = cv2.FONT_HERSHEY_PLAIN

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def get_gaze_ratio(frame, eye_points, facial_landmarks):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                               np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    down_side_threshold = threshold_eye[int(height/2):height, 0:width]
    down_side_white = cv2.countNonZero(down_side_threshold)

    up_side_threshold = threshold_eye[0:int(height/2), 0:width]
    up_side_white = cv2.countNonZero(up_side_threshold)

    horizontal_view = ""

    if abs(left_side_white - right_side_white) < 50:
        horizontal_view = "CENTER"
    elif left_side_white > right_side_white:
        horizontal_view = "LEFT"
    elif right_side_white > left_side_white:
        horizontal_view = "RIGHT"

    vertical_view = ""

    if 10 < abs(up_side_white - down_side_white) < 40:
        vertical_view = "CENTER"
    elif abs(up_side_white - down_side_white) == 0:
        vertical_view = "DOWN"
    elif down_side_white > up_side_white:
        vertical_view = "UP"

    return horizontal_view, vertical_view


def main():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:

            landmarks = predictor(gray, face)

            horizontal, vertical = get_gaze_ratio(frame, [36, 37, 38, 39, 40, 41], landmarks)

            if vertical == horizontal:
                cv2.putText(frame, "CENTER", (50, 300), font, 2, (0, 0, 255), 3)
            elif vertical == "DOWN" and horizontal == "CENTER":
                cv2.putText(frame, "DOWN", (50, 300), font, 2, (0, 0, 255), 3)
            elif vertical == "UP" and horizontal == "CENTER":
                cv2.putText(frame, "UP", (50, 300), font, 2, (0, 0, 255), 3)
            elif horizontal == "LEFT" and vertical == "CENTER":
                cv2.putText(frame, "LEFT", (50, 300), font, 2, (0, 0, 255), 3)
            elif horizontal == "RIGHT" and vertical == "CENTER":
                cv2.putText(frame, "RIGHT", (50, 300), font, 2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, horizontal + " " + vertical, (50, 300), font, 2, (0, 0, 255), 3)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

