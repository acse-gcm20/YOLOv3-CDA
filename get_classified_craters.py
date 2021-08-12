import pandas as pd
import os
from tqdm import tqdm

def ids_from_file(filename):
    ids = []

    with open(filename) as f:
        for line in f.readlines():
            line = line.rstrip('\n').split(' ')
            if line[5] != '':
                ids.append(line[5])

    return ids

def get_lat_long(idx):

    v1_id = id_df.iloc[idx]['v1']

    row = v1df[v1df['CRATER_ID']==v1_id].iloc[0]

    lat = row['LATITUDE_CIRCLE_IMAGE']
    lon = row['LONGITUDE_CIRCLE_IMAGE']

    id_df.iloc[idx]['latitude'] = lat
    id_df.iloc[idx]['longitude'] = lon

def get_v2(idx):

    lat = id_df.iloc[idx]['latitude']
    lon = id_df.iloc[idx]['longitude']

    try:
        row = v2df[(v2df['LATITUDE_CIRCLE_IMAGE']==lat) & (v2df['LONGITUDE_CIRCLE_IMAGE']==lon)].iloc[0]
        v2_id = row['CRATER_ID']
        id_df.iloc[idx]['v2'] = v2_id
        return 0
    except IndexError:
        return 1

def get_deg_state(idx):

    v2_id = id_df.iloc[idx]['v2']

    row = v2df[v2df['CRATER_ID']==v2_id].iloc[0]
    ds = row['DEGRADATION_STATE']

    if ds is not '':
        id_df.iloc[idx]['degradation_state'] = ds

v1 = 'data/Robbins/Robbins_v1.csv'
v2 = 'data/Robbins/Robbins_v2.csv'

v1df = pd.read_csv(v1)[['CRATER_ID', 'LATITUDE_CIRCLE_IMAGE', 'LONGITUDE_CIRCLE_IMAGE']]
v2df = pd.read_csv(v2, low_memory=False)[['CRATER_ID', 'LATITUDE_CIRCLE_IMAGE', 'LONGITUDE_CIRCLE_IMAGE', 'DEGRADATION_STATE']]

labels = 'data/Robbins/labels/'

crater_ids = []
missed_craters = []

for label_file in os.listdir(labels):
    crater_ids += ids_from_file(labels+label_file)

id_df = pd.DataFrame({'v1':crater_ids, 'v2':'', 'latitude':'', 'longitude':'', 'degradation_state':''})

for i in tqdm(range(len(id_df))):
    get_lat_long(i)
    if get_v2(i) == 1:
        missed_craters.append(i)
    else:
        get_deg_state(i):

print(f'missed {len(missed_craters)} craters')
print(missed_craters)
print(id_df.head(), '\n')

id_df.to_csv('id_dictionary.csv')
