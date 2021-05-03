import os, time, csv, torch, cv2
import torchvision
import asyncio
import numpy as np
import pandas as pd
from models import HandNNModel, classify_arbitrary_image, classify_many_images, asl_prediction_to_class_str, prediction_to_class_str
from HandFinder import HandFinder
from dataset_manipulation import reshape_img

async def classify_img(model, img, pred_fn):
    window_name = classify_arbitrary_image(model, img, pred_to_str_fn=pred_fn)
    return window_name


async def show_classified_hands(model, hands, pred_fn):
    for idx, hand in enumerate(hands):
        try:
            resized_hand = reshape_img(hand, (256,256))
            resized_hand = np.array([resized_hand])
            window_name = await asyncio.gather(classify_img(model, resized_hand, pred_fn))
            cv2.imshow(window_name[0], hand)
        except cv2.error as e:
            print(str(e))

def run_demo(model, pred_fn):
    cap = cv2.VideoCapture(0)
    hand_finder = HandFinder()
    while True:
        ret, frame = cap.read()
        imgs, hands = hand_finder.extractHands(frame)
        cv2.imshow('frame', frame)
        asyncio.run(show_classified_hands(model, hands, pred_fn))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_alexnet_model():
    alexnet_model = torchvision.models.alexnet(pretrained=True)
    alexnet_model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2))
    alexnet_model.classifier[6] = torch.nn.Linear(4096, 26)
    return alexnet_model

def main():
    device = torch.device('cuda')
    alexnet_model = get_alexnet_model()
    print(alexnet_model.eval())
    save_path = "../data/asl/Asl_AdamNet_tl.pkl"
    print(alexnet_model.load_state_dict(torch.load(save_path)))
    alexnet_model.eval()
    alexnet_model.cuda(device=device)
    run_demo(alexnet_model, asl_prediction_to_class_str)
    #loaded_model = HandNNModel()
    #loaded_model.load_state_dict(torch.load(save_path))
    #loaded_model = loaded_model.to(device=device)
    #loaded_model.eval()
    #run_demo(loaded_model)

if __name__ == "__main__":
    main()