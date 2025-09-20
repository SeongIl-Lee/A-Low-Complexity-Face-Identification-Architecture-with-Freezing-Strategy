import os
import cv2
import numpy as np
import pandas as pd
from dataset import *
from test import *
from mtcnn import MTCNN




# 1) Extract images from video
def save_frames_from_video(video_path, output_dir):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS)) 
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  
    print(f"FPS: {fps}, 총 프레임 수: {frame_count}")
    success = True
    count = 0
    frame_index = 0
    while success:
        success, image = video.read()
        if success:
            file_name = os.path.join(output_dir, f'image_{count:04d}.jpg')
            cv2.imwrite(file_name, image)
            print(f'{count}번째 이미지 저장 완료: {file_name}')
            count += 1
        frame_index += 1
    video.release()
    print(f"총 {count}개의 이미지가 저장되었습니다.")
    return count


# 2) Crop the face region using mtcnn
def crop_faces_mtcnn(path):
    listdir_perclass = os.listdir(path)
    class_num = 0

    for cl in listdir_perclass:
        count = 0
        img_dir_perclass = os.path.join(path, cl)
        list_dir_perclass_perimg = os.listdir(img_dir_perclass)
        for sample in list_dir_perclass_perimg:
            img_path = os.path.join(img_dir_perclass, sample)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            detector = MTCNN()
            detections = detector.detect_faces(img_rgb)
            for i, face in enumerate(detections):
                x, y, width, height = face['box']
                cropped_face = img[y:y+height, x:x+width]  
                save_path = f'{class_num}_%04d.jpg' % count
                cv2.imwrite(save_path, cropped_face)
                print(f"Saved cropped face {i} to {save_path}")
                count += 1
        class_num += 1


# 3) Create a CSV file containing cropped face image information(path, class name, class id)
def create_dataset_csv_file(img_path):
    idx = 0
    df = pd.DataFrame(columns=['idx','path','class_name','class_id'])
    class_name_paths = ['class0','class1','class2']
    for i in range(len(class_name_paths)):
        class_name_path = os.path.join(img_path,class_name_paths[i]) 
        per_dirs = os.listdir(class_name_path) 
        for j in range(len(per_dirs)):
            total_path = os.path.join(class_name_path,per_dirs[j]) 
            class_name = class_name_paths[i]
            class_id = int(class_name[-1])
            data = pd.Series({'idx': idx, 'path':total_path, 'class_name': class_name, 'class_id':class_id})
            data_df = pd.DataFrame([data])
            df = pd.concat([df, data_df], ignore_index=True)
            print(f'인덱스:{idx}, class_name : {class_name}, class_id : {class_id} 현재 개수 : {j}')
            idx += 1
    df.to_csv(f'dataframe/custom_dataset_front_side_face/joint/task1/class_02.csv')
    
    
# 4) Split  total images to train/test/val images
def split_train_val_test(df):
    class_ids = df.loc[:,'class_id'].unique() 

    train_indices_list = []
    validation_indices_list = []
    test_indices_list = []

    for i in class_ids:
        
        df_idx_perclass = df.loc[(df.class_id == i),'idx'].values
        
        train_ratio = 0.6
        validation_ratio = 0.2
        
        dataset_size = len(df_idx_perclass)
        train_size = int(dataset_size * train_ratio)
        validation_size = int(dataset_size * validation_ratio)

        indices= np.random.permutation(df_idx_perclass)
        train_indices = indices[:train_size]
        validatiaon_indices = indices[train_size:train_size+validation_size]
        test_indices = indices[train_size+validation_size:]

        train_indices_list.append(train_indices)
        validation_indices_list.append(validatiaon_indices)
        test_indices_list.append(test_indices)
        
    train_indices_total = np.concatenate(train_indices_list)
    validation_indices_total = np.concatenate(validation_indices_list)
    test_indices_total = np.concatenate(test_indices_list)
    
    train_df = df.loc[df['idx'].isin(train_indices_total),:]
    train_df['idx'] = range(len(train_df))
    train_df = train_df.drop(columns=['Unnamed: 0'])
    
    validation_df = df.loc[df['idx'].isin(validation_indices_total),:]
    validation_df['idx'] = range(len(validation_df))
    validation_df = validation_df.drop(columns=['Unnamed: 0'])

    test_df = df.loc[df['idx'].isin(test_indices_total),:]
    test_df['idx'] = range(len(test_df))
    test_df = test_df.drop(columns=['Unnamed: 0'])
    
    train_df.to_csv(f'dataframe/train.csv')
    validation_df.to_csv(f'dataframe/val.csv') 
    test_df.to_csv(f'dataframe/test.csv')



