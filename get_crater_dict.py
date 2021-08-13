import pandas as pd
import os
from tqdm import tqdm

class ID_dict:

    def __init__(self, v1, v2, crater_ids):
        self.v1df = pd.read_csv(v1)[['CRATER_ID', 'LATITUDE_CIRCLE_IMAGE', 'LONGITUDE_CIRCLE_IMAGE']]
        self.v2df = pd.read_csv(v2, low_memory=False)[['CRATER_ID', 'LATITUDE_CIRCLE_IMAGE', 'LONGITUDE_CIRCLE_IMAGE', 'DEGRADATION_STATE']]
        self.id_df = pd.DataFrame({'v1':crater_ids, 'v2':'', 'latitude':'', 'longitude':'', 'degradation_state':''})

    def get_lat_long(self, idx):
        """Method to find the lat and long of a crater from its version 1 ID"""
        
        # Get version 1 ID from index and extract row
        v1_id = self.id_df.iloc[idx]['v1']
        row = self.v1df[self.v1df['CRATER_ID']==v1_id].iloc[0]

        # Get lat and long and save to dataframe
        lat = row['LATITUDE_CIRCLE_IMAGE']
        lon = row['LONGITUDE_CIRCLE_IMAGE']
        self.id_df.iloc[idx]['latitude'] = lat
        self.id_df.iloc[idx]['longitude'] = lon

    def get_v2(self, idx):
        """Method to find the v2 ID from the lat and long of a crater"""
        lat = self.id_df.iloc[idx]['latitude']
        lon = self.id_df.iloc[idx]['longitude']

        try:
            # Look for lat/long in version 2 dataframe and extract the version 2 ID
            row = self.v2df[(self.v2df['LATITUDE_CIRCLE_IMAGE']==lat) & (self.v2df['LONGITUDE_CIRCLE_IMAGE']==lon)].iloc[0]
            v2_id = row['CRATER_ID']
            # Add version 2 ID inplace
            self.id_df.iloc[idx]['v2'] = v2_id
            return 0
        except IndexError:
            # If cannot find a lat/long match return an error
            return 1

    def get_deg_state(self, idx):
        """Method to lookup the degradation state of a crater"""

        # Get version 2 ID from index and extract degradation state of crater
        v2_id = self.id_df.iloc[idx]['v2']
        row = self.v2df[self.v2df['CRATER_ID']==v2_id].iloc[0]
        ds = row['DEGRADATION_STATE']

        # If there is a degradation state, add it to dataframe
        if ds != '':
            self.id_df.iloc[idx]['degradation_state'] = ds

def ids_from_file(filename):
    """Method to extract list of IDs of craters in a file"""
    ids = []

    with open(filename) as f:
        for line in f.readlines():
            line = line.rstrip('\n').split(' ')

            if line[5] != '': # non-Robbins craters do not have an ID
                ids.append(line[5])

    return ids

def run():
    # Version 1 of Robbins database
    # Contains the IDs used in labels of processed Robbins data
    v1 = 'data/Robbins/Robbins_v1.csv'

    # Version 2 of Robbins database
    # Contains degradation state information and different ID convention
    v2 = 'data/Robbins/Robbins_v2.csv'

    # Path to directory containing all the labels of processed Robbins images
    labels = 'data/Robbins/crater_group_dataset/labels/'

    crater_ids = []
    missed_craters = []

    for label_file in os.listdir(labels):
        # Get list of version 1 IDs of all craters in processed images
        crater_ids += ids_from_file(labels+label_file)
    
    print(f'{len(os.listdir(labels))} files')
    print(f'{len(crater_ids)} craters')

    # Generate dataframe to hold:
    # version 1 ID, version 2 ID, latitude, longitude, degradation state
    # of all processed craters
    id_dict = ID_dict(v1, v2, crater_ids)

    # For every crater...
    for i in tqdm(range(len(id_dict.id_df))):
        # Add lat and long to dataframe
        id_dict.get_lat_long(i)
        # Add version 2 ID to dataframe where possible
        if id_dict.get_v2(i) == 1:
            missed_craters.append(i)
        else:
            # If found version 2 ID, add degradation state if possible
            id_dict.get_deg_state(i)

    print(f'{len(id_dict.id_df)} craters saved')
    print(f'{len(missed_craters)} craters missed')
    print(id_dict.id_df.head(), '\n')

    id_dict.id_df.to_csv('crater_dictionary.csv')
