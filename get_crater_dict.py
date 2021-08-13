import pandas as pd
import os
from tqdm import tqdm

class ID_dict:

    def __init__(self, v1, v2, crater_ids):
        self.v1df = pd.read_csv(v1)[['CRATER_ID', 'LATITUDE_CIRCLE_IMAGE', 'LONGITUDE_CIRCLE_IMAGE']]
        self.v2df = pd.read_csv(v2, low_memory=False)[['CRATER_ID', 'LATITUDE_CIRCLE_IMAGE', 'LONGITUDE_CIRCLE_IMAGE', 'DEGRADATION_STATE']]
        self.id_df = pd.DataFrame({'v1':crater_ids, 'v2':'', 'latitude':'', 'longitude':'', 'degradation_state':''})

    def get_lat_long(self, idx):

        v1_id = self.id_df.iloc[idx]['v1']

        row = self.v1df[self.v1df['CRATER_ID']==v1_id].iloc[0]

        lat = row['LATITUDE_CIRCLE_IMAGE']
        lon = row['LONGITUDE_CIRCLE_IMAGE']

        self.id_df.iloc[idx]['latitude'] = lat
        self.id_df.iloc[idx]['longitude'] = lon

    def get_v2(self, idx):

        lat = self.id_df.iloc[idx]['latitude']
        lon = self.id_df.iloc[idx]['longitude']

        try:
            row = self.v2df[(self.v2df['LATITUDE_CIRCLE_IMAGE']==lat) & (self.v2df['LONGITUDE_CIRCLE_IMAGE']==lon)].iloc[0]
            v2_id = row['CRATER_ID']
            self.id_df.iloc[idx]['v2'] = v2_id
            return 0
        except IndexError:
            return 1

    def get_deg_state(self, idx):

        v2_id = self.id_df.iloc[idx]['v2']

        row = self.v2df[self.v2df['CRATER_ID']==v2_id].iloc[0]
        ds = row['DEGRADATION_STATE']

        if ds != '':
            self.id_df.iloc[idx]['degradation_state'] = ds

def ids_from_file(filename):
    ids = []

    with open(filename) as f:
        for line in f.readlines():
            line = line.rstrip('\n').split(' ')
            if line[5] != '':
                ids.append(line[5])

    return ids

def run():
    v1 = 'data/Robbins/Robbins_v1.csv'
    v2 = 'data/Robbins/Robbins_v2.csv'

    labels = 'data/Robbins/crater_group_dataset/labels/'

    crater_ids = []
    missed_craters = []

    for label_file in os.listdir(labels):
        crater_ids += ids_from_file(labels+label_file)
    
    print(f'{len(os.listdir(labels))} files')
    print(f'{len(crater_ids)} craters')

    id_dict = ID_dict(v1, v2, crater_ids)

    for i in tqdm(range(len(id_dict.id_df))):
        id_dict.get_lat_long(i)
        if id_dict.get_v2(i) == 1:
            missed_craters.append(i)
        else:
            id_dict.get_deg_state(i)

    print(f'{len(id_dict.id_df)} craters saved')
    print(f'{len(missed_craters)} craters missed')
    print(id_dict.id_df.head(), '\n')

    id_dict.id_df.to_csv('crater_dictionary.csv')
