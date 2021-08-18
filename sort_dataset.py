import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm

def get_filename(latitude, longitude):
    """Filenames are in format {lat}_{long} (.png/.txt) where lat 
    and long refer to the top-left corner of the image. Each image 
    is 1 degree by 1 degree in size """
    f_lat = int(latitude)
    f_long = int(longitude)

    if latitude > 0:
        f_lat += 1
    
    if longitude < 0:
        f_long -= 1

    name = f'{f_lat}_{f_long}'

    return name

def write_image_list(imgs, dest):
    """Create a list of images containing classified craters from 
    a Dataframe. This list can be used in Colab to transfer the correct files"""

    pth = f'{dest}/image_list.txt'
    
    with open(pth, 'w') as f:
        for _, row in imgs.iterrows():
            filename = row['filename']
            f.write(filename+'.png\n')

class Dataset:
    """Sort a dataset in preparation for training a classifier"""
    def __init__(self, crater_dict_csv, loss_csv, threshold, dest,
                 source_dir, clean=False):
        self.crater_dict_csv = crater_dict_csv # Path to csv file of all craters
        self.loss_csv = loss_csv    # Path to csv file containing losses
        self.threshold = threshold  # desired object loss threshold
        self.dest = dest # Destination directory e.g. 'data/Robbins/classifier'
        self.stats = stats  # Print statisics about dataset
        self.clean = clean
        self.source_dir = source_dir # Directory of original labels

        # Create and clean the necessary directories
        if not os.path.exists(self.dest):
            os.makedirs(self.dest)
        else:
            shutil.rmtree(self.dest)

        if not os.path.exists(f'{self.dest}/images/'):
            os.makedirs(f'{self.dest}/images/')
        else:
            shutil.rmtree(f'{self.dest}/images/')

        if not os.path.exists(f'{self.dest}/labels/'):
            os.makedirs(f'{self.dest}/labels/')
        else:
            shutil.rmtree(f'{self.dest}/labels/')

        # Get Dataframe of all classified craters and list of filenames containing them
        self.craters, self.filenames = self.generate_df()

        # Get good_imgs dataframe of images below loss threshold
        self.good_imgs = self.sort_obj_loss()

        # Write good images to list file
        write_image_list(self.good_imgs, self.dest)

        if self.clean:
            # Create secondary list of images exclusively containing classified craters
            self.clean_list()
            self.sort_files(f'{self.dest}/clean_image_list.txt')
        else:
            self.sort_files(f'{self.dest}/image_list.txt')

        self.analyse()

    def generate_df(self):
        # Read csv containing all processed craters
        data = pd.read_csv(self.crater_dict_csv)

        # Remove craters without deg_state and remove duplicates
        craters = data[data['degradation_state'].notnull()].drop_duplicates(subset='v1').drop('Unnamed: 0', axis=1)

        # Reset the index
        craters.reset_index(drop=True, inplace=True)

        # list of filenames from lat/long
        filenames = [get_filename(row["latitude"], row["longitude"]) for _, row in craters.iterrows()]

        # add filenames column to dataframe
        craters['filename'] = np.array(filenames)

        craters.to_csv('./data/Robbins/classified_craters.csv')

        return craters, list(set(filenames))

    def sort_obj_loss(self):
        # Ranking of images by objectness loss
        if self.threshold == 1:
            # If threshold is 1, use entire dataset
            good_imgs = pd.DataFrame({'filename':self.filenames,
                                      'obj_loss':np.zeros(len(self.filenames))})
        else:
            data = pd.read_csv(self.loss_csv)

            # Generate dataframe of the images we want to use and store their objectness
            class_data = pd.DataFrame({'filename':self.filenames,
                                       'obj_loss':np.zeros(len(self.filenames))})

            cnt = 0
            for i, row in class_data.iterrows():
                #print(row['filename'])
                try:
                    loss = data[data['img']==row['filename']]['obj'].iloc[0]
                    class_data.loc[i, 'obj_loss'] = loss
                except:
                    print(f'Cannot find {row["filename"]} in loss rank')
                    class_data.loc[i, 'obj_loss'] = 1
                    cnt+=1            

            class_data.sort_values('obj_loss', ascending=True, inplace=True)
            class_data.reset_index(drop=True, inplace=True)

            # Generate 'good_imgs' dataframe of images above desired objectness threshold
            threshold_loss = class_data.iloc[int(len(class_data) * self.threshold)]['obj_loss']
            good_imgs = class_data[class_data['obj_loss'] < threshold_loss]
            print('Missed', cnt)
        return good_imgs

    def clean_list(self):

        # Open list of good images
        with open(f'{self.dest}/image_list.txt', 'r') as img_list:
            imgs = img_list.readlines()

        filenames = [img.rstrip('.png\n') for img in imgs]

        # Create new clean image list file
        clean_img_file = open(f'{self.dest}/clean_image_list.txt', 'w')

        cnt = 0
        for filename in tqdm(filenames):
            with open(f'{self.source_dir}/labels/{filename}.txt') as label_file:
                labels = label_file.readlines()
            
            states = []
            good = True
            while good:
                for label in labels:
                    label = label.rstrip('\n').split(' ')
                    # Check all labels have a crater ID
                    if label[-1] == '':
                        good = False
                        break
                    else:
                        # Add degradation states to list
                        crater_id = label[-1]
                        print('Crater ID:', crater_id)
                        deg_state = self.craters[self.craters['v1']==crater_id]['degradation_state'].iloc[0]
                        states.append(deg_state)
                if good and True not in np.isnan(states):
                    # Only accept label files which exclusively contain classified craters
                    # Write image name to clean list file
                    clean_img_file.write(f'{filename}.png\n')
                    cnt +=1
                    break

        print(f'\n{cnt} images in {self.dest}/clean_image_list.txt\n')
        clean_img_file.close()

    def sort_files(self, img_list_path):
        # Copy the relevant images and labels from processed Robbins data
        # Remove unclassified craters and correct class labels
        with open(img_list_path, 'r') as img_list:
            files = [img.rstrip('.png\n') for img in img_list.readlines()]

        for filename in tqdm(files):
            # Copy image to destination directory
            os.system(f'cp {self.source_dir}/images/{filename}.png {self.dest}/images/')

            # Create new label in classifier directory
            new_label = open(f'{self.dest}/labels/{filename}.txt', 'w')
            # Collect IDs of craters in file
            ids = self.craters[self.craters['filename']==filename]['v1']

            # Loop through craters in image and write new label for classified craters
            # Remove labels of unclassified craters
            label_pth = f'{self.source_dir}/labels/{filename}.txt'
            with open(label_pth) as f:
                lines = f.readlines()

            for line in lines:
                line = line.rstrip('\n').split(' ')

                # if original crater and has degradation state
                # replace label with classification and write to new label file
                if len(line) == 6 and line[-1] in ids.values:
                    new_line = line
                    row = self.craters[self.craters['v1'] == line[-1]].iloc[0]
                    state = (int(row['degradation_state']) - 1) # Subtract one to zero index labels
                    new_line = ' '.join([str(state)] + line[1:]) + '\n'

                    with open(f'{self.dest}/labels/{filename}.txt', 'w') as new_label:
                        new_label.write(new_line)

    def analyse(self):
        # Print dataset information 
        if self.clean:
            img_list = f'{self.dest}/clean_image_list.txt'
        else:
            img_list = f'{self.dest}/image_list.txt'

        print(f'\n--- Evaluating {img_list} ---\n')

        with open(img_list, 'r') as f:
            imgs = f.readlines()

        imgs = [img.rstrip(' \n') for img in imgs]
        num_imgs = len(imgs)

        fnames = [fname.split('.')[0] for fname in imgs]
        num_lbls = 0
        for fname in fnames:
            with open(f'{self.source_dir}/labels/{fname}.txt') as f:
                num_lbls += len(f.readlines())

        print(f'Images: {num_imgs}\nLabels: {num_lbls}')


