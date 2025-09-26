import os 
import numpy as np 
import random 
import pandas as pd 

random.seed(758)


def standardize_data(data:pd.DataFrame):
    # z-score normalization
    alpha = 0.01
    df = data 

    allnpy = df[['L_Acc_X', 'L_Acc_Y', 'L_Acc_Z', 'R_Acc_X', 'R_Acc_Y', 'R_Acc_Z']].to_numpy()
    allmean = allnpy.mean()
    allstd = allnpy.std() + alpha

    df['L_Acc_X'] = (df['L_Acc_X'] - allmean) / allstd
    df['L_Acc_Y'] = (df['L_Acc_Y'] - allmean) / allstd
    df['L_Acc_Z'] = (df['L_Acc_Z'] - allmean) / allstd
    df['R_Acc_X'] = (df['R_Acc_X'] - allmean) / allstd
    df['R_Acc_Y'] = (df['R_Acc_Y'] - allmean) / allstd
    df['R_Acc_Z'] = (df['R_Acc_Z'] - allmean) / allstd
    
    allnpy = df[['L_Gyr_X', 'L_Gyr_Y', 'L_Gyr_Z', 'R_Gyr_X', 'R_Gyr_Y', 'R_Gyr_Z']].to_numpy()
    allmean = allnpy.mean()
    allstd = allnpy.std() + alpha

    df['L_Gyr_X'] = (df['L_Gyr_X'] - allmean) / allstd
    df['L_Gyr_Y'] = (df['L_Gyr_Y'] - allmean) / allstd
    df['L_Gyr_Z'] = (df['L_Gyr_Z'] - allmean) / allstd
    df['R_Gyr_X'] = (df['R_Gyr_X'] - allmean) / allstd
    df['R_Gyr_Y'] = (df['R_Gyr_Y'] - allmean) / allstd
    df['R_Gyr_Z'] = (df['R_Gyr_Z'] - allmean) / allstd

    return df 

def normalize_data(data:pd.DataFrame):
    df = data 

    cols = list(df.columns.unique())

    for col in cols:
        dfmax, dfmin = df[col].max(), df[col].min()
        df[col] = (df[col] - dfmin) / (dfmax - dfmin)

    return df 

# csv -> 3d npy
def make_3d_imu_npy(data:pd.DataFrame, standardize=True, normalize=True):
    if standardize:
        data = standardize_data(data)
    if normalize:
        data = normalize_data(data)

    X_list = ['L_Acc_X', 'L_Gyr_X', 'R_Acc_X', 'R_Gyr_X']
    Y_list = ['L_Acc_Y', 'L_Gyr_Y', 'R_Acc_Y', 'R_Gyr_Y']
    Z_list = ['L_Acc_Z', 'L_Gyr_Z', 'R_Acc_Z', 'R_Gyr_Z']

    sample_X = data.loc[:, X_list]
    sample_Y = data.loc[:, Y_list]
    sample_Z = data.loc[:, Z_list]

    sample_X = sample_X.to_numpy()
    sample_Y = sample_Y.to_numpy()
    sample_Z = sample_Z.to_numpy()

    data3d = np.dstack([sample_X, sample_Y, sample_Z])

    return data3d


# csv -> 1d npy
def make_1d_imu_npy(data:pd.DataFrame, standardize=True, normalize=False):
    usecols = ['L_Acc_X', 'L_Acc_Y', 'L_Acc_Z', 'L_Gyr_X', 'L_Gyr_Y', 'L_Gyr_Z', 'R_Acc_X', 'R_Acc_Y', 'R_Acc_Z', 'R_Gyr_X', 'R_Gyr_Y', 'R_Gyr_Z']
    data = data.loc[:, usecols]

    if standardize:
        data = standardize_data(data)
    if normalize:
        data = normalize_data(data)
    data1d = data.to_numpy()

    return data1d 



# sliding window setting
WINDOW_SIZE = 120
STRIDE = 10
# path of segmented data
DATA_ROOT = './data/export/imu_segment'

# sliding window 
def sliding_window_with_label(data, lab, window_size=WINDOW_SIZE, stride=STRIDE, start_point=0):
    window_size = window_size 
    strd = stride 
    start_point = start_point
    label = lab
    window_li = []

    while start_point + window_size <= len(data):
        window = data[start_point:start_point+window_size]
        start_point = start_point + strd 
        window_li.append([window, label])

    # random.shuffle(window_li)
    sw_res = window_li
    return sw_res

# data list -> sliding window per data -> merge all
def many_sliding_window_all(dpaths, labs, depth='3d'):
    dcols = ['L_Acc_X', 'L_Acc_Y', 'L_Acc_Z', 'L_Gyr_X', 'L_Gyr_Y', 'L_Gyr_Z', 'R_Acc_X', 'R_Acc_Y', 'R_Acc_Z', 'R_Gyr_X', 'R_Gyr_Y', 'R_Gyr_Z']
    merged_sw_res = {'data':[], 'label': []}
    before_shuffle = []
    
    for dpath, lab in zip(dpaths, labs):
        if 'norm_' not in dpath: 
            dpath = 'norm_' + dpath # norm_ 추가
        data = pd.read_csv(os.path.join(DATA_ROOT, dpath), index_col=None, usecols=dcols)
        if len(data) == 0: continue
        if depth == '3d':
            data = make_3d_imu_npy(data, True, False)
        elif depth == '1d':
            data = make_1d_imu_npy(data)
        window_res = sliding_window_with_label(data, lab)
        before_shuffle = before_shuffle + window_res 

    # random.shuffle(before_shuffle)

    for windowlab in before_shuffle:
        window = windowlab[0]
        lab = windowlab[1]
        merged_sw_res['data'].append(window)
        merged_sw_res['label'].append(lab)

        # for window, label in zip(window_res['data'], window_res['label']):
        #    merged_sw_res['data'].append(window)
        #    merged_sw_res['label'].append(label)

    return merged_sw_res


# data list -> sliding window per data -> merge random 1 window
def many_sliding_window(dpaths, labs, depth='3d'):
    dcols = ['L_Acc_X', 'L_Acc_Y', 'L_Acc_Z', 'L_Gyr_X', 'L_Gyr_Y', 'L_Gyr_Z', 'R_Acc_X', 'R_Acc_Y', 'R_Acc_Z', 'R_Gyr_X', 'R_Gyr_Y', 'R_Gyr_Z']
    merged_sw_res = {'data':[], 'label': []}
    before_shuffle = []
    
    for dpath, lab in zip(dpaths, labs):
        if 'norm_' in dpath: 
            dpath = dpath[5:] # norm_ 제거
        data = pd.read_csv(os.path.join(DATA_ROOT, dpath), index_col=None, usecols=dcols)
        if len(data) == 0: continue
        if depth == '3d':
            data = make_3d_imu_npy(data, True, False)
        elif depth == '1d':
            data = make_1d_imu_npy(data)
        window_res = sliding_window_with_label(data, lab)

        if len(window_res) == 0:
            continue 
        chosen_window = random.choice(window_res)

        before_shuffle.append(chosen_window)

    # random.shuffle(before_shuffle)
    print("imu windows", len(before_shuffle))

    for windowlab in before_shuffle:
        window = windowlab[0]
        lab = windowlab[1]
        merged_sw_res['data'].append(window)
        merged_sw_res['label'].append(lab)

        # for window, label in zip(window_res['data'], window_res['label']):
        #    merged_sw_res['data'].append(window)
        #    merged_sw_res['label'].append(label)

    return merged_sw_res