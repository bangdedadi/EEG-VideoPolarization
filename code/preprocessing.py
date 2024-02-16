# %%
import json
# from statistics import median_grouped
import pandas as pd
# from pyrsistent import v
import mne
import os
import time
import tqdm
import math
import glob

# %%
def dic_v2k(trigger2event_dict):
    re = {}
    for k,v in trigger2event_dict.items():
        re[v] = k
    return re

# %%
def load_txt(file_name):
    trigger2trigger2time = {}
    with open(file_name) as f:
        for line in f.readlines():
            t1, t2, time = line.strip().split(':')
            if t1 not in trigger2trigger2time.keys():
                trigger2trigger2time[t1] = {}
            trigger2trigger2time[t1][t2] = time
    return trigger2trigger2time

# %%
def timestamp_convert_localdate(timestamp,time_format="%Y/%m/%d %H:%M:%S"):
    # Convert according to the current device's timezone, for example, Beijing time UTC+8
    timeArray = time.localtime(timestamp)
    styleTime = time.strftime(str(time_format), timeArray)
    return styleTime

# %%
def filter_255(events_from_annot):
    re = []
    for item in events_from_annot:
        if item[-1] != 255:
            re.append(item)
    return re

# %%
# [Extract the subject's rating information]
def load_data_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)
user_data = load_data_from_file('user_data_full.json')

# %%
id_map

# %%
# List of characters (in order)
renwu=['Edison','Cao Cao','Columbus','Emperor Wu of Han','Catherine','Lao Bai','Mike','Newton','Qin Shi Huang','Xiang Yu']
# Function to load Excel and add 'tend' information based on the character list 'renwu' and user_data
def load_excel_with_tend(file_name, user_data, renwu, pinyin_map):
    df = pd.read_excel(file_name)
    df.sort_values("local_time_ms", inplace=True, ignore_index=True)

    char_to_index_map = {char: i for i, char in enumerate(renwu)}

    v2info = {}
    for i in range(len(df)):
        current_id = str(df.loc[i]['item_id'])  # Convert to string to prevent JSON serialization issues

        if current_id not in v2info:
            v2info[current_id] = {'video_type': 0, 'start_time': -1, 'end_time': -1, 'prev_start_time': -1, 'tend': 2}

        current_like = int(df.loc[i]['video_type'])  # Convert using int()
        v2info[current_id]['video_type'] = current_like

        if df.loc[i]['event'] == 'video_play':
            v2info[current_id]['prev_start_time'] = v2info[current_id]['start_time']
            v2info[current_id]['start_time'] = int(df.loc[i]['local_time_ms'])  # Convert using int()

        elif df.loc[i]['event'] in ['video_end', 'nextone']:
            v2info[current_id]['end_time'] = int(df.loc[i]['local_time_ms'])  # Convert using int()
            v2info[current_id]['prev_start_time'] = -1

        if v2info[current_id]['start_time'] > v2info[current_id]['end_time'] and v2info[current_id]['prev_start_time'] != -1:
            v2info[current_id]['start_time'] = v2info[current_id]['prev_start_time']

        char_index = None
        for char in renwu:
            if char in current_id:
                char_index = char_to_index_map[char]
                break

        if char_index is not None:
            for name, pinyin in pinyin_map.items():
                if pinyin in str(df['会员账号'].values[0]):  # Convert to string
                    for this_user_data in user_data:
                        if this_user_data['name'] == name:
                            v2info[current_id]['tend'] = this_user_data['tend'][char_index]
                            break
                    break

    return v2info

# %%
class Transformer:
    def __init__(self,):
        self.b = 0
        self.k = 0
    def fit(self, action, event):
        self.k = (event[1] - event[0]) / (action[1] - action[0])
        self.b = event[1] - action[1] * self.k
        return self.k, self.b
    def action2event(self,action):
        return int(self.k * action + self.b + 0.5)      

# %%
def find_time(start_time, time2t2t, time_stamps, txt_info):
    if start_time < time_stamps[0] or start_time > time_stamps[-1]:
        return None
    for i, time_stamp in enumerate(time_stamps):
        # mising three triggers
        if time_stamps[i+1] - time_stamp > 10000 * 3:
            return None
        if start_time >= time_stamp and start_time < time_stamps[i+1]:
            real_time = [time_stamp, time_stamps[i+1]]
            eeg_time = []
            for rtime in real_time:
                t1, t2 = time2t2t[rtime]
                eeg_time.append(txt_info[t1][t2]['eeg_time'])
            transformer = Transformer()
            transformer.fit(real_time, eeg_time)
            if abs(transformer.k - 1000) > 50:
                with open('tmp.txt','a') as f:
                    f.write(str(real_time))
                    f.write('\t')
                    f.write(str(eeg_time))
                    f.write('\t')
                    f.write(str(t1)+'\t'+str(t2))
                    f.write('\n')
                print(transformer.k)
                return None
            return transformer.action2event(start_time)

# %%
def map_info(txt_info, events_from_annot, excel_info):
    
    event_group = []
    for i in range(len(events_from_annot)):
        if i + 1 < len(events_from_annot):
            time_diff = events_from_annot[i+1][0] - events_from_annot[i][0]
            if time_diff < 1050 and time_diff > 950:
                event_group.append([events_from_annot[i][2], events_from_annot[i+1][2], events_from_annot[i+1][0]])
                
                event_key = str(events_from_annot[i][2])
                next_event_key = str(events_from_annot[i+1][2])

                if event_key in txt_info and next_event_key in txt_info[event_key]:
                    current_value = txt_info[event_key][next_event_key]

                    if isinstance(current_value, (int, float, str)):
                        try:
                            float_value = float(current_value)
                            txt_info[event_key][next_event_key] = {
                                'time': float_value,
                                'eeg_time': float(events_from_annot[i+1][0])
                            }
                        except ValueError:
                            print(f"Warning: Cannot convert {current_value} to float.")
                    else:
                        print(f"Warning: Value at txt_info[{event_key}][{next_event_key}] is not a number or string, it's a {type(current_value)}")
    time2t2t = {}
    for t1 in txt_info.keys():
        for t2 in txt_info[t1].keys():
            if type(txt_info[t1][t2]) == dict:
                time2t2t[txt_info[t1][t2]['time']] = [t1, t2]
    time_stamps = list(sorted([float(item) for item in time2t2t.keys()]))

    v2info = excel_info
    for v in v2info.keys():
        start_time = int(v2info[v]['start_time'] / 1e3)
        end_time = int(v2info[v]['end_time'] / 1e3)
        time_diff = int(end_time - start_time)
        eeg_time = find_time(start_time, time2t2t, time_stamps, txt_info)
        if eeg_time != None:
            v2info[v]['eeg_start_time'] = int(eeg_time)
        eeg_time = find_time(end_time, time2t2t, time_stamps, txt_info)
        if eeg_time != None:
            v2info[v]['eeg_end_time'] = int(eeg_time)
    return v2info

# %%
for date in tqdm.tqdm([
           'participant_1','participant_2'

]):
    student_id = date
    print(f'-----------------------------preprocessing data of user {student_id}-------------------------------')
    file_name = date + '.cnt'
        
    raw = mne.io.read_raw_cnt('./lab2-data/'+file_name, preload=True, verbose='WARNING')
    # tmp_file_name = '../data/eeg/' + str(student_id) + '_1 Data.cnt'
    # if os.path.exists(tmp_file_name):
    #     raw_0 = mne.io.read_raw_cnt(tmp_file_name, preload=True, verbose='WARNING')
    #     raw.append(raw_0)

    channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]

    raw.pick_channels(channels)

    events_from_annot, event_dict = mne.events_from_annotations(raw, verbose='WARNING')
    #print(events_from_annot.shape)
    event2trigger_dic = dic_v2k(event_dict)
    for idx in range(len(events_from_annot)):
        events_from_annot[idx][2] = event2trigger_dic[events_from_annot[idx][2]]

    txt_info = load_txt('./lab2-txt/'+date+'.txt')
    #print(txt_info)
    #excel_info = load_excel('./lab1-log/'+date+'.xlsx')
    excel_info = load_excel_with_tend('./lab2-log/'+date+'.xlsx',user_data, renwu, pinyin_map)
    eeg_data = raw.get_data()
    events_from_annot = filter_255(events_from_annot)

    idx2eeg = {}
    idx = 0
    v2info = map_info(txt_info, events_from_annot, excel_info)

    for v in v2info.keys():
        v2info[v]['start_time'] = int(v2info[v]['start_time'])
        v2info[v]['end_time'] = int(v2info[v]['end_time'])
        if 'eeg_start_time' not in v2info[v].keys() or 'eeg_end_time' not in v2info[v].keys():
            continue
        if v2info[v]['eeg_end_time'] - v2info[v]['eeg_start_time'] < 0:
            print('error',v)
            continue
        time_diff = min(v2info[v]['eeg_end_time'] - v2info[v]['eeg_start_time'], 60 * 1000)
        idx2eeg[idx] = eeg_data[:,v2info[v]['eeg_start_time']:v2info[v]['eeg_start_time']+time_diff].tolist()
        v2info[v]['idx'] = int(idx)
        idx += 1

    json.dump(v2info, open('./v2info/'+date+'_v2info.json','w'))
    json.dump(idx2eeg, open('./x2eeg/'+date+'_idx2eeg.json','w'))


