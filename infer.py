import sys
import os
import requests
import argparse
import time

import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from util.compare_pixel import compare_images_pixel, compare_images_pixel_ssim, compare_images_pixel_value, process_detections, prepare_mae_model, prepare_vit_model,get_top_match_category,reconstruct_image, save_to_txt, get_unique_dir, get_knn_match_category, get_weight_match_category
from util.eval import evaluate_predictions
from util.build_memory_bank import build_memory_bank

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def get_args_parser():
    parser = argparse.ArgumentParser('infer', add_help=False)
    parser.add_argument('--device', default='cuda', type=str, help='')
    parser.add_argument('--images_dir', default='dataset/images', type=str, help='')
    parser.add_argument('--mae_model',default='mae_vit_base_patch14', type=str, help='')
    parser.add_argument('--checkpoint_path', default='checkpoint-399.pth', type=str, help='')
    parser.add_argument('--build_memory_bank', action='store_true', help='')
    parser.add_argument('--memory_bank_path', default='memory_bank/memory_bank_10.pth', type=str, help='')
    parser.add_argument('--support_images_dir', default='memory_bank/support_images_10', type=str, help='')
    parser.add_argument('--results_dir', default='data/detect', type=str, help='')
    parser.add_argument('--re_times', default=1, type=int, help='reconstruction times')
    parser.add_argument('--mask_ratio', default=0.25, type=float, help='')
    parser.add_argument('--knn', default=3, type=int, help='KNN')
    parser.add_argument('--eval', action='store_true', help='')
    parser.add_argument('--save_txt', action='store_true', help='save the result txt files')
    return parser



def main(args):
    if args.build_memory_bank:
        build_memory_bank(folder_path=args.support_images_dir, save_path=args.memory_bank_path, model_name=args.mae_model[4:], 
                      checkpoint=args.checkpoint_path, 
                      device='cuda', input_size=224)
    
    images_dir = args.images_dir
    save_images_dir = get_unique_dir(args.results_dir)
    if not os.path.exists(save_images_dir):
        os.makedirs(save_images_dir)
        
    chkpt_path=args.checkpoint_path
    mae_model=prepare_mae_model(chkpt_path, arch=args.mae_model, device=args.device)
    vit_model=prepare_vit_model(chkpt_path, model_name=args.mae_model[4:], global_pool=True, device=args.device)
    memory_bank=torch.load(args.memory_bank_path)
        
    image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(('.png'))]
    k=args.re_times
    start_time=time.time()
    for img_path in tqdm(image_paths, desc="Processing images"):
    
        original_img = Image.open(img_path)
        img = original_img.resize((224, 224))
        img = np.array(img) / 255.

        assert img.shape == (224, 224, 3)

        img = img - imagenet_mean
        img = img / imagenet_std
       
        res=[]
 
        for _ in range(k):
            original_, visible_, randomMask = reconstruct_image(img, mae_model, mask_ratio=args.mask_ratio)

            result=compare_images_pixel_value(original_,visible_)
            contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tmp=[]
            for contour in contours:
                tmp.append(cv2.boundingRect(contour))
            res.append(tmp)
      
        intersections=process_detections(res,k)
        
        result_img=np.array(original_img.resize((224, 224)))

        detections = []

        for intersection in intersections:
            x,y,w,h=map(round,intersection)  
            roi = result_img[y:y+h, x:x+w]
            category=get_top_match_category(input_image=roi, vit_model=vit_model, memory_bank=memory_bank, input_size=224, device='cuda')
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{category}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, 1)
            if y - text_height - 2 < 0:
                text_start_y = y + text_height
            else:
                text_start_y = y - 2
            cv2.rectangle(result_img, (x, text_start_y - text_height - 3), (x + text_width, text_start_y), (0, 255, 0), -1)
            cv2.putText(result_img, label, (x, text_start_y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            detections.append(map(str,[category, round((x+0.5*w)/224,6), round((y+0.5*h)/224,6), round(w/224,6), round(h/224,6)]))
            
        save_image_dir=os.path.join(save_images_dir, os.path.basename(img_path).replace('.png', ''))
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)   
        cv2.imwrite(save_image_dir+'/original.png', original_)
        for i, im_masked in enumerate(randomMask):    
            cv2.imwrite(save_image_dir+'/randomMask_{}.png'.format(i+1), im_masked)  
        cv2.imwrite(save_image_dir+'/visible.png', visible_)
        cv2.imwrite(save_image_dir+'/mask.png', result)  
        cv2.imwrite(save_image_dir+'/result.png', result_img)
        
        if args.save_txt:
            save_txt_dir=save_images_dir+'/labels'
            if not os.path.exists(save_txt_dir):
                    os.makedirs(save_txt_dir)
            if not detections:
                detections.append(map(str,[12,0.5,0.5,1.0,1.0]))
            save_to_txt(os.path.join(save_txt_dir, os.path.basename(img_path).replace('.png', '.txt')), detections)
    end_time=time.time()
    average_processing_time =(end_time-start_time)/ len(image_paths) 
    print(f"Average processing time per image: {average_processing_time:.2f} seconds")      
     
    if args.eval:
        evaluate_predictions(
        pred_labels_path=args.results_dir+'/labels',
        gt_folder='dataset/labels',
        gt_coco_file="coco/coco_gt.json",
        pred_coco_file="coco/coco_dt.json",
        class_id_to_name={1.0:'1', 2.0:'2', 3.0:'3', 4.0:'4', 5.0:'5'}
    )
        
            
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

