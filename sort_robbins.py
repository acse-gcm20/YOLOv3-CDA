import numpy as np
import pandas as pd
import os
import shutil

def get_filename(latitude, longitude):

    f_lat = int(latitude)
    f_long = int(longitude)

    if latitude > 0:
        f_lat += 1
    
    if longitude < 0:
        f_long -= 1

    name = f'{f_lat}_{f_long}'

    return name

def generate_df(pth, save_csv=True):

    data = pd.read_csv(pth)

    # Remove craters without deg_state and remove duplicates
    craters = data[data['degradation_state'].notnull()].drop_duplicates(subset='v1').drop('Unnamed: 0', axis=1)

    # Reset the index
    craters.reset_index(drop=True, inplace=True)

    # list of filenames from lat/long
    filenames = [get_filename(row["latitude"], row["longitude"]) for _, row in craters.iterrows()]

    # add filenames column to dataframe
    craters['filename'] = np.array(filenames)

    if save_csv:
        craters.to_csv('./data/Robbins/classified_craters.csv')

    return craters, filenames

def write_image_list(imgs):
    # Create a list of images containing classified craters
    # This list can be used in Colab to transfer the correct files
    pth = './data/Robbins/classifier/image_list'
    
    with open(pth, 'w') as f:
        for _, row in imgs.iterrows():
            filename = row['filename']
            f.write(filename+'.png/n')

def sort_obj_loss(filenames, threshold):
    # Rank objectness loss of images with classified craters
    pth = './data/Robbins/loss_rank.csv'

    data = pd.read_csv(pth)
    class_data = pd.DataFrame({'filename':filenames, 'obj_loss':np.zeros(len(filenames))})
    class_data.drop_duplicates(subset='filename', inplace=True)

    for i, row in class_data.iterrows():
        loss = data[data['img']==row['filename']]['obj'].iloc[0]
        class_data.loc[i, 'obj_loss'] = loss

    class_data.sort_values('obj_loss', ascending=True, inplace=True)
    class_data.reset_index(drop=True, inplace=True)

    if threshold == 1:
        good_imgs = class_data
    else:
        threshold_loss = class_data.iloc[int(len(class_data) * threshold)]['obj_loss']
        good_imgs = class_data[class_data['obj_loss'] < threshold_loss]

    return good_imgs

def sort_files(files, craters):
    # Copy the relevant labels from processed Robbins data
    # Remove unclassified craters and correct class labels 
    for filename in files:
        pth = f'./data/Robbins/labels/{filename}.txt'
        new_label = open(f'./data/Robbins/classifier/labels/{filename}.txt', 'w')
        ids = craters[craters['filename']==filename]['v1']

        with open(pth) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('/n').split(' ')

                # if original crater and in deg_state df
                if len(line) == 6 and line[-1] in ids.values:
                    new_line = line
                    row = craters[craters['v1'] == line[-1]].iloc[0]
                    state = int(row['degradation_state'])
                    new_line = ' '.join([str(state)] + line[1:]) + '/n'

                    
                    new_label.write(new_line)
        
        new_label.close()

def analyze(craters, imgs, threshold):
    print(f'Crater Distribution (threshold = {threshold})')
    df = craters[craters['filename'].isin(imgs['filename'])]
    print(df['degradation_state'].value_counts().sort_index().astype(int))
    print(f'Total craters: {len(df)}')

def main(deg_state_csv, threshold, stats=True):

    craters, files = generate_df(deg_state_csv)
    good_imgs = sort_obj_loss(files, threshold)
    write_image_list(good_imgs)

    if stats:
        analyze(craters, good_imgs, threshold)
