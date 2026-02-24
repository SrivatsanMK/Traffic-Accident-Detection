import cv2
import numpy as np
import os
from keras.models import model_from_json

# ==========================================
# 1. THE MODEL CLASS
# ==========================================
class AccidentDetectionModel(object):
    class_nums = ['Accident', "No Accident"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_accident(self, img):
        self.preds = self.loaded_model.predict(img)
        return AccidentDetectionModel.class_nums[np.argmax(self.preds)], self.preds


# ==========================================
# 2. THE MAIN LOGIC
# ==========================================
def startapplication():
    # Initialize the model and font
    model = AccidentDetectionModel("model/model.json", "model\model_weights.h5")
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Open the video (change to 0 for webcam)
    video = cv2.VideoCapture('videos/01940.mp4')

    # Windows Sise
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 1000, 600)

    while True:
        ret, frame = video.read()
        
        # Safety check: if the video ends or frame can't be read, stop the loop
        if not ret:
            print("Video ended or cannot be read.")
            break

        # Image preprocessing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        # Get prediction from the model
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        
        if pred == "Accident":
            prob = (round(prob[0][0] * 100, 2))
            
            # to beep when alert:
            # if(prob > 90):
            #     os.system("say beep")

            # Draw the alert on the screen
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred + " " + str(prob) + "%", (20, 30), font, 1, (255, 255, 0), 2)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        cv2.imshow('Video', frame)

    # Clean up windows and release the video after quitting
    video.release()
    cv2.destroyAllWindows()


# ==========================================
# 3. THE TRIGGER
# ==========================================
if __name__ == '__main__':
    startapplication()