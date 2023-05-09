import os
import face_recognition, cv2
from defaults import *
import pickle
from time import time


class Trainer:
    def __init__(self, camera: int = 0):
        self.known_face_encodings = []
        self.camera = camera
        self.known_face_names = []

    def get_training(self, _path: str):
        for name in os.listdir(_path):
            if os.path.isfile(name):
                continue
            path = os.path.join("TrainingSet", name)
            for pics in os.listdir(path):
                if os.path.isfile(pics) and not pics.endswith(".pickle"):
                    image_of_person = face_recognition.load_image_file(
                        path + "/" + pics
                    )
                    person_face_encoding = face_recognition.face_encodings(
                        image_of_person
                    )[0]
                    self.known_face_encodings.append(person_face_encoding)
                    self.known_face_names.append(name)
                else:
                    with open(path, "rb") as f:
                        self.known_face_encodings = pickle.load(f)
                        self.known_face_names = os.listdir(path)
        return self.known_face_encodings, self.known_face_names

    def train(self):
        if not os.path.exists(path):  # Place where all the training data will be stored
            os.mkdir(path)
        pics_to_save = []
        camera = cv2.VideoCapture(self.camera)
        while True:
            ret, frame = camera.read()
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                pics_to_save.append(frame)
                print(f"Picture taken...... for [ {len(pics_to_save)} ]")
                if len(pics_to_save) >= 5:
                    break
        camera.release()
        cv2.destroyAllWindows()
        name = input("Enter the name of the person: ")
        remove_index = list()
        for i, pics in enumerate(pics_to_save):
            cv2.imshow("Save this pictures?", pics)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("s"):  # save the picture
                path_to_save = os.path.join(path, name)  # this will be the directory
            else:
                print("Not saving the picture... :/")
                print()
                remove_index.append(i)
        for i in remove_index[::-1]:  # removing the pictures that are not to be saved
            pics_to_save.pop(i)
        self.save(self.create_encoding(pics_to_save), path_to_save)
        self.displayAsciiArt(name=name)

    def save(self, encodings, path_to_save):
        if os.path.exists(path_to_save):  # Duplicate name...
            path_to_save += f"_{int(time())}"
        os.mkdir(path_to_save)
        for i, encoding in enumerate(encodings, 1):
            with open(path_to_save + f"/encoding_{i}.pickle", "wb") as f:
                pickle.dump(encoding, f)

    def create_encoding(self, frames):
        results = []
        for i, frame in enumerate(frames, 1):
            try:
                results.append(face_recognition.face_encodings(frame)[0])
            except IndexError:
                print(f"No face found in the image number.... {i}")
            else:
                print(f"Face found in the image number.... {i} [ OK ]")
        return results

    def displayAsciiArt(self, name):
        print("\n\n  _______")
        print(" /       \\")
        print("/  O   O  \\")
        print("|    ^    |")
        print("|   '-'   |")
        print(" \\_______/\n\n")
        print(f"Welcome {name}! You have been added to the database.")
        print()
