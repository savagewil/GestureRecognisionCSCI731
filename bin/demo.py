import os, time, csv, torch, cv2
import numpy as np
import pandas as pd
from models import HandNNModel, classify_arbitrary_image, classify_many_images
from HandFinder import HandFinder
from dataset_manipulation import reshape_img

def run_demo(model):
    cap = cv2.VideoCapture(0)
    hand_finder = HandFinder()
    while True:
        ret, frame = cap.read()
        imgs, hands = hand_finder.extractHands(frame)
        cv2.imshow('frame', frame)
        for idx, hand in enumerate(hands):
            try:
                resized_hand = reshape_img(np.array([hand]), (256,256))
                # TODO use async to let this run, b/c it's slow af
                window_name = classify_arbitrary_image(model, resized_hand)
                cv2.imshow(window_name, hand)
            except Exception as e:
                print(str(e))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    device = torch.device('cuda')
    save_path = "../deep_trained_cnn.pkl"
    loaded_model = HandNNModel()
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model = loaded_model.to(device=device)
    run_demo(loaded_model)

if __name__ == "__main__":
    main()