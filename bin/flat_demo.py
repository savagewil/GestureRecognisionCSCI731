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


# async def show_classified_hands(model, hands, pred_fn):
#     for idx, hand in enumerate(hands):
#         try:
#             resized_hand = reshape_img(hand, (256,256))
#             resized_hand = np.array([resized_hand])
#             window_name = await asyncio.gather(classify_img(model, resized_hand, pred_fn))
#             cv2.imshow(window_name[0], hand)
#         except cv2.error as e:
#             print(str(e))

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

def run_demo(model, pred_fn, image_path, flip=False):
    hand_finder = HandFinder()

    frame = cv2.imread(image_path)
    if flip:
        frame = cv2.flip(frame, 1)
    imgs, hands, hand_rects = hand_finder.extractHands(frame)
    cv2.imshow('frame', frame)
    labels = classify_hands(model, hands, pred_fn)

    for rect, label in zip(hand_rects, labels):
        x,y,w,h = rect
        cv2.rectangle(frame, (x,y),(x+w, y+h), (255,0,0),5)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, 35, thickness=5), (255,0,0), thickness=5)

    while True:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return frame

def get_alexnet_model():
    alexnet_model = torchvision.models.alexnet(pretrained=True)
    alexnet_model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2))
    alexnet_model.classifier[6] = torch.nn.Linear(4096, 26)
    return alexnet_model

def main(image_path, out_path, flip=False):
    device = torch.device('cuda')
    alexnet_model = get_alexnet_model()
    print(alexnet_model.eval())
    save_path = "../data/asl/Asl_AdamNet_tl.pkl"
    print(alexnet_model.load_state_dict(torch.load(save_path)))
    alexnet_model.eval()
    alexnet_model.cuda(device=device)
    labeled = run_demo(alexnet_model, asl_prediction_to_class_str, image_path, flip)
    cv2.imwrite(out_path+os.path.basename(image_path), labeled)
    print("Saved image to "+out_path+os.path.basename(image_path))
    #loaded_model = HandNNModel()
    #loaded_model.load_state_dict(torch.load(save_path))
    #loaded_model = loaded_model.to(device=device)
    #loaded_model.eval()
    #run_demo(loaded_model)

if __name__ == "__main__":
    image_path = sys.argv[1]

    flip = len(sys.argv) >= 3 and  "-f" in sys.argv
    save_path = "../images/FlatDemoLabeled/"
    main(image_path, save_path, flip)