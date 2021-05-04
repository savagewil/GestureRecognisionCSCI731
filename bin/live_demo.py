import os, time, csv, torch, cv2
import sys

import torchvision
import asyncio
import numpy as np
import pandas as pd
from models import HandNNModel, classify_arbitrary_image, classify_many_images, asl_prediction_to_class_str, prediction_to_class_str
from HandFinder import HandFinder
from dataset_manipulation import reshape_img

def classify_img(model, img, pred_fn):
    window_name = classify_arbitrary_image(model, img, pred_to_str_fn=pred_fn)
    return window_name

def classify_hands(model, hands, pred_fn):
    labels = []
    for idx, hand in enumerate(hands):
        try:
            resized_hand = reshape_img(hand, (256, 256))
            resized_hand = np.array([resized_hand])
            label = classify_img(model, resized_hand, pred_fn)
            labels.append(label.upper())
        except cv2.error as e:
            print(str(e))
    return labels

def run_demo(model, pred_fn, save_path = None, flip=False):
    cap = cv2.VideoCapture(0)
    if save_path is not None:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        videoWriter = cv2.VideoWriter(save_path+str(time.time()) + '.avi',
                                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                      60,
                                      (frame_width, frame_height))
    hand_finder = HandFinder()
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if flip:
            frame = cv2.flip(frame, 1)
        imgs, hands, hand_rects = hand_finder.extractHands(frame)
        labels = classify_hands(model, hands, pred_fn)
        for rect, label in zip(hand_rects, labels):
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, 35, thickness=5), (255, 0, 0), thickness=5)
        cv2.imshow('frame', frame)
        if save_path is not None:
            videoWriter.write(frame)
        print(time.time() - start_time)
        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    cap.release()
    if save_path is not None:
        videoWriter.release()
    cv2.destroyAllWindows()

def get_alexnet_model():
    alexnet_model = torchvision.models.alexnet(pretrained=True)
    alexnet_model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2))
    alexnet_model.classifier[6] = torch.nn.Linear(4096, 26)
    return alexnet_model

def main(out_path, flip):

    device = torch.device('cuda')
    alexnet_model = get_alexnet_model()
    print(alexnet_model.eval())
    save_path = "../data/asl/Asl_AdamNet_tl.pkl"
    print(alexnet_model.load_state_dict(torch.load(save_path)))
    alexnet_model.eval()
    alexnet_model.cuda(device=device)
    run_demo(alexnet_model, asl_prediction_to_class_str, out_path, flip)
    #loaded_model = HandNNModel()
    #loaded_model.load_state_dict(torch.load(save_path))
    #loaded_model = loaded_model.to(device=device)
    #loaded_model.eval()
    #run_demo(loaded_model)

if __name__ == "__main__":

    flip = "-f" in sys.argv
    record = "-r" in sys.argv
    save_path = "../images/LiveDemoCaptures/" if record else None
    main(save_path, flip)