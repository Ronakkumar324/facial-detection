from defaults import *
import cv2, face_recognition
import pickle
import numpy as np
import threading


class Detector:
    def __init__(self, camera: int = 0, trained_models: str = path) -> None:
        self.camera = camera
        if not os.path.exists(trained_models):
            os.mkdir(trained_models)
            print("No trained models found. Please train the model first.")
            print("Can't recognize faces without a trained model....")
            print()
            return
        self.names = os.listdir(trained_models)
        self.trained_model = []
        names = self.names.copy()
        for i, name in enumerate(names):
            file_names = os.listdir(os.path.join(path, name))
            trigger = True
            for file_name in file_names:
                if file_name.endswith(".pickle"):
                    with open(os.path.join(path, name, file_name), "rb") as f:
                        self.trained_model.append(pickle.load(f))
                    if trigger:
                        trigger = False
                    else:
                        self.names.insert(i, name)

            if name.find("_") != -1:
                self.names[i] = name[: name.find("_")]
        if len(self.names) != len(self.trained_model):
            print("Error loading trained models")
            return
        self.trained_model = np.array(self.trained_model)
        self.names = np.array(self.names)
        self.frame = None
        self.stop_event = threading.Event()

    def start_detection(self):
        video_capture = cv2.VideoCapture(self.camera)
        print("Starting detection....")
        print(self.names, self.trained_model)

        def read_frames():
            while not self.stop_event.is_set():
                ret, frame = video_capture.read()
                if ret:
                    self.original_frame = frame
                    self.frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        read_thread = threading.Thread(target=read_frames)
        process_frame = True

        read_thread.start()
        while True:
            if self.frame is None:
                continue
            small_frame = self.frame.copy()
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            if process_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations
                )
                face_names = []

                for face_encoding in face_encodings:
                    # matches = face_recognition.compare_faces(
                    #     self.trained_model, face_encoding
                    # )
                    # name = "Unknown"
                    # if any(matches):
                    #     first_match_index = matches.index(True)
                    #     name = self.names[first_match_index]
                    # face_names.append(name)
                    distances = face_recognition.face_distance(
                        self.trained_model, face_encoding
                    )
                    if np.any(distances < 0.3):
                        index = np.argmin(distances)
                        name = self.names[index]
                    else:
                        name = "Unknown"
                    face_names.append(name)
            process_frame = not process_frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(
                    self.original_frame, (left, top), (right, bottom), (0, 0, 255), 2
                )
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    self.original_frame,
                    name,
                    (left + 6, bottom - 6),
                    font,
                    1.0,
                    (255, 255, 255),
                    1,
                )
            cv2.imshow("Video", self.original_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.stop_event.set()
        read_thread.join()
        video_capture.release()
        cv2.destroyAllWindows()
