import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm

def get_filename(latitude, longitude):

    # Filenames are in format {lat}_{long} (.png/.txt)
    # where lat and long refer to the top-left corner of the image
    # Each image is 1 degree by 1 degree in size
    f_lat = int(latitude)
    f_long = int(longitude)

    if latitude > 0:
        f_lat += 1
    
    if longitude < 0:
        f_long -= 1

    name = f'{f_lat}_{f_long}'

    return name

def generate_df(pth, save_csv=True):

    # Read csv containing all processed craters
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

    return craters, list(set(filenames))

def write_image_list(imgs):
    # Create a list of images containing classified craters
    # This list can be used in Colab to transfer the correct files
    pth = 'data/Robbins/classifier/image_list.txt'
    
    with open(pth, 'w') as f:
        for _, row in imgs.iterrows():
            filename = row['filename']
            f.write(filename+'.png\n')

def sort_obj_loss(filenames, threshold):
    # Ranking of images by objectness loss
    if threshold == 1:
        # If threshold is 1, use entire dataset
        good_imgs = pd.DataFrame({'filename':filenames, 'obj_loss':np.zeros(len(filenames))})
    else:
        data = pd.read_csv('data/Robbins/loss_rank.csv')

        # Generate dataframe of the images we want to use and store their objectness
        class_data = pd.DataFrame({'filename':filenames, 'obj_loss':np.zeros(len(filenames))})
        #class_data.drop_duplicates(subset='filename', inplace=True)

        for i, row in class_data.iterrows():
            loss = data[data['img']==row['filename']]['obj'].iloc[0]
            class_data.loc[i, 'obj_loss'] = loss

        class_data.sort_values('obj_loss', ascending=True, inplace=True)
        class_data.reset_index(drop=True, inplace=True)

        # Generate 'good_imgs' dataframe of images above desired objectness threshold
        threshold_loss = class_data.iloc[int(len(class_data) * threshold)]['obj_loss']
        good_imgs = class_data[class_data['obj_loss'] < threshold_loss]

    return good_imgs

def clean_list(csv_path):
    # Load crater dictionary into dataframe
    crater_dict = pd.read_csv(csv_path)

    # Open list of good images
    with open('data/Robbins/classifier/image_list.txt', 'r') as img_list:
        imgs = img_list.readlines()

    filenames = [img.rstrip('.png\n') for img in imgs]

    # Create new clean image list file
    clean_img_file = open('data/Robbins/classifier/clean_image_list.txt', 'w')

    cnt = 0
    for filename in tqdm(filenames):
        with open(f'data/Robbins/crater_group_dataset/labels/{filename}.txt') as label_file:
            labels = label_file.readlines()
        
        states = []
        good = True
        #print(f'File: {filename}')
        while good:
            # Check lengths to verify all craters are original
            for label in labels:
                label = label.rstrip('\n').split(' ')
                if label[-1] == '':
                    good = False
                    break
                else:
                    id = label[-1]
                    ds = crater_dict[crater_dict['v1']==id]['degradation_state'].iloc[0]
                    states.append(ds)
            if good:
                if True not in np.isnan(states):
                    clean_img_file.write(f'{filename}.png\n')
                    cnt +=1
                break
            else:
                #print('Bad')
                good = False

    print(f'\n{cnt} images in data/Robbins/classifier/clean_image_list.txt')
    clean_img_file.close()

def sort_files(craters, img_list_path):
    # Copy the relevant images and labels from processed Robbins data
    # Remove unclassified craters and correct class labels
    with open(img_list_path, 'r') as img_list:
        files = [img.rstrip('.png\n') for img in img_list.readlines()]

        for filename in tqdm(files):
            # Copy image to classifier directory
            os.system(f'cp data/Robbins/crater_group_dataset/images/{filename}.png data/Robbins/classifier/images/')

            # Create new label in classifier directory
            new_label = open(f'./data/Robbins/classifier/labels/{filename}.txt', 'w')
            # Collect IDs of craters in file
            ids = craters[craters['filename']==filename]['v1']

            # Loop through craters in image and write new label for classified craters
            # Remove unclassified labels
            label_pth = f'./data/Robbins/crater_group_dataset/labels/{filename}.txt'
            with open(label_pth) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.rstrip('\n').split(' ')

                    # if original crater and has degradation state
                    # replace label with classification and write to new label file
                    if len(line) == 6 and line[-1] in ids.values:
                        new_line = line
                        row = craters[craters['v1'] == line[-1]].iloc[0]
                        state = (int(row['degradation_state']) - 1) # Subtract one to zero index labels
                        new_line = ' '.join([str(state)] + line[1:]) + '\n'

                        with open(f'./data/Robbins/classifier/labels/{filename}.txt', 'w') as new_label:
                            new_label.write(new_line)
            
def analyze(craters, imgs, threshold):
    # Get dataframe of all classified craters in the desired images
    df = craters[craters['filename'].isin(imgs['filename'])]

    # Print dataset information 
    print(f'Crater Distribution (threshold = {threshold})')
    print(df['degradation_state'].value_counts().sort_index().astype(int))
    print(f'Total craters: {len(df)}')
    print(f'Total images: {len(imgs)}')

def main(crater_dict, threshold, stats=True, clean=True):

    if not os.path.exists('data/Robbins/classifier'):
        os.makedirs('data/Robbins/classifier')
    else:
        shutil.rmtree('data/Robbins/classifier/')

    if not os.path.exists('data/Robbins/classifier/images/'):
        os.makedirs('data/Robbins/classifier/images/')
    else:
        shutil.rmtree('data/Robbins/classifier/images/')

    if not os.path.exists('data/Robbins/classifier/labels/'):
        os.makedirs('data/Robbins/classifier/labels/')
    else:
        shutil.rmtree('data/Robbins/classifier/labels/')

    # os.mkdirs('data/Robbins/classifier')
    # os.mkdirs('data/Robbins/classifier/images/')
    # os.mkdirs('data/Robbins/classifier/labels/')


    # Generate dataframe of classified craters and list of filenames containing them
    craters, files = generate_df(crater_dict)

    # Sort images by objectness and return df of images above threshold
    good_imgs = sort_obj_loss(files, threshold)

    # Write the list of desired images to a file
    # This file is used in Colab to transfer the dataset
    write_image_list(good_imgs)

    # Transfer desired image and label files to separate classifier directory
    if clean:
        print('\nCleaning\n')
        clean_list('data/Robbins/crater_dictionary.csv')
        sort_files(craters, 'data/Robbins/classifier/clean_image_list.txt')
    else:
        sort_files(craters, 'data/Robbins/classifier/image_list.txt')

    # Print stats
    if stats:
        analyze(craters, good_imgs, threshold)
