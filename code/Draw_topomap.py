# %%
import mne
import json
import scipy.stats
import copy
import numpy as np
import matplotlib.pyplot as plt

# %%
montage = mne.channels.read_dig_fif('montage.fif')
montage.ch_names = json.load(open("montage_ch_names.json"))
montage.dig = montage.dig[:64]
montage.ch_names = montage.ch_names[:64]
for i in range(len(montage.dig)):
    montage.dig[i]['r'] = np.array([item * 1e-6 for item in montage.dig[i]['r']])
ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
for dig_info_ in ten_twenty_montage.dig:
    dig_info = copy.deepcopy(dig_info_)
    if 'EEG' not in str(dig_info['kind']):
        montage.dig.insert(0, dig_info)
picked_channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2"]
total_channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2", ]
fake_info = mne.create_info(ch_names=total_channels, sfreq=1000., ch_types='eeg')
select_index = [idx for idx in range(len(total_channels)) if total_channels[idx] in picked_channels]

# %%
multi_data = {
    'subject':[],
    'positive_env':[],
    'negative_env':[],
    'SEED_video':[],
    'neutral':[],
    'fear':[],
    'sad':[],
    'happy':[],
    'po_env_po_video':[],
    'po_env_ne_video':[],
    'ne_env_po_video':[],
    'ne_env_ne_video':[],
    'po_env_high_va':[],
    'po_env_low_va':[],
    'ne_env_high_va':[],
    'ne_env_low_va':[],
    'po_env_high_ro':[],
    'po_env_low_ro':[],
    'ne_env_high_ro':[],
    'ne_env_low_ro':[],
    'po_env_with_like':[],
    'po_env_without_like':[],
    'ne_env_with_like':[],
    'ne_env_without_like':[],
    'po_env_high_time':[],
    'po_env_low_time':[],
    'ne_env_high_time':[],
    'ne_env_low_time':[]
}

def add_features(data_list, features):
    # Add the feature array (3, 62, 5) to the list
    for feature in features:
        data_list.append(feature)
for date in ['LAB2-huqifan', 'LAB2-cangyueyang', 'LAB2-hongyurui', 'LAB2-fanhao', 'LAB2-dongyimeng', 'LAB2-houlinzhi', 'LAB2-jiwenjun', 'LAB2-lujianing', 'LAB2-miaoshengze',
             'LAB2-wanfangwei', 'LAB2-wangxiaoting', 'LAB2-wangzhengni', 'LAB2-yangchen', 'LAB2-zhangxue', 'LAB2-liangqihang', 'LAB2-daisiwei',
             'LAB2-zhangyutong', 'LAB2-mengfanjie', 'LAB2-zhangchenxi', 'LAB2-liangyanshu', 'LAB2-zhaochensong', 'LAB2-chenrong', 'LAB2-chenxingyu']:
    v2info = json.load(open('./v2info/' + date + '_v2info.json'))
    idx2de = json.load(open('./hope_features/' + date + '_idx2de.json'))
    for v in v2info.keys():
        if 'idx' not in v2info[v].keys():
            continue
        features = np.array(idx2de[str(v2info[v]['idx'])])
        if features.shape == (30, 62, 5):
            # Divide the 30 features into 3 groups, each with 10 features
            grouped_features = features.reshape(1, 30, 62, 5).mean(axis=1)
            #grouped_features = features.mean(axis=0)
            # Add to the corresponding list based on the conditions of tend and video_type
            if v2info[v]['tend'] == 0:
                add_features(multi_data['positive_env'], grouped_features)
                add_features(multi_data['subject'], grouped_features)
                if v2info[v]['video_type'] == 0:
                    add_features(multi_data['po_env_po_video'], grouped_features)
                if v2info[v]['video_type'] == 1:
                    add_features(multi_data['po_env_ne_video'], grouped_features)
                if v2info[v]['valence']>5:
                    add_features(multi_data['po_env_high_va'], grouped_features)
                if v2info[v]['valence']<5:
                    add_features(multi_data['po_env_low_va'], grouped_features)
                if v2info[v]['arousal']>5:
                    add_features(multi_data['po_env_high_ro'], grouped_features)
                if v2info[v]['arousal']<5:
                    add_features(multi_data['po_env_low_ro'], grouped_features)
                if v2info[v]['like'] == 1:
                    add_features(multi_data['po_env_with_like'], grouped_features)
                if v2info[v]['like'] == 0:
                    add_features(multi_data['po_env_without_like'], grouped_features)
                if v2info[v]['play_duration'] >0.9:
                    add_features(multi_data['po_env_high_time'], grouped_features)
                if v2info[v]['play_duration'] <0.9:
                    add_features(multi_data['po_env_low_time'], grouped_features)
            elif v2info[v]['tend'] == 1:
                add_features(multi_data['negative_env'], grouped_features)
                add_features(multi_data['subject'], grouped_features)
                if v2info[v]['video_type'] == 0:
                    add_features(multi_data['ne_env_po_video'], grouped_features)
                if v2info[v]['video_type'] == 1:
                    add_features(multi_data['ne_env_ne_video'], grouped_features)
                if v2info[v]['valence']>5:
                    add_features(multi_data['ne_env_high_va'], grouped_features)
                if v2info[v]['valence']<5:
                    add_features(multi_data['ne_env_low_va'], grouped_features)
                if v2info[v]['arousal']>5:
                    add_features(multi_data['ne_env_high_ro'], grouped_features)
                if v2info[v]['arousal']<5:
                    add_features(multi_data['ne_env_low_ro'], grouped_features)
                if v2info[v]['like'] == 1:
                    add_features(multi_data['ne_env_with_like'], grouped_features)
                if v2info[v]['like'] == 0:
                    add_features(multi_data['ne_env_without_like'], grouped_features)
                if v2info[v]['play_duration'] >0.9:
                    add_features(multi_data['ne_env_high_time'], grouped_features)
                if v2info[v]['play_duration'] <0.9:
                    add_features(multi_data['ne_env_low_time'], grouped_features)
            elif v2info[v]['tend'] == 2:
                add_features(multi_data['SEED_video'], grouped_features)
                if v2info[v]['video_type'] == 2:
                    add_features(multi_data['neutral'], grouped_features)
                elif v2info[v]['video_type'] == 3:
                    add_features(multi_data['sad'], grouped_features)
                elif v2info[v]['video_type'] == 4:
                    add_features(multi_data['fear'], grouped_features)
                elif v2info[v]['video_type'] == 5:
                    add_features(multi_data['happy'], grouped_features)
# Convert lists to numpy arrays
for key in multi_data.keys():
    multi_data[key] = np.array(multi_data[key])

# %%
significance = np.ones((62,5))
diff = np.zeros((62,5))
for channel in range(62):
    for band in range(5):
        # Adjust the correlation of different variables in the EEG topomap as needed
        like_list = multi_data['po_env_low_time'][:,channel,band]
        dislike_list = multi_data['ne_env_low_time'][:,channel,band]

        y = [0 for i in range(len(dislike_list))] + [1 for i in range(len(like_list))]
        x = dislike_list.tolist() + like_list.tolist()
        r, pval = scipy.stats.pearsonr(x, y)
        significance[channel,band] = pval
        diff[channel,band] = r

fake_evoked = mne.EvokedArray(diff, fake_info).pick_channels(picked_channels)
fake_evoked.set_montage(montage)

data = fake_evoked.data
data = np.array(data)

# Modify subplot settings to create 5 subplots instead of 6
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 4), gridspec_kw=dict(top=0.9), sharex=True, sharey=True,dpi=200)

vmin = -0.1
vmax = 0.1

for idx in range(5):
    fs_data = data[:, idx]
    mask = significance[select_index, idx] < 0.05
    item = mne.viz.plot_topomap(fs_data, fake_evoked.info, axes=ax[idx], show=False, vmin=vmin, vmax=vmax, mask=mask, mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=3), cmap='seismic',outlines='head',extrapolate='head')
colorbar=fig.colorbar(item[0], ax=ax[-1])
# Set the y-axis scale
new_ticks = [-0.1, -0.05, 0,0.05,0.1]  # Adjust the scale as needed
colorbar.set_ticks(new_ticks)
# Set the font size of coordinate numbers
colorbar.ax.tick_params(labelsize=12)  # Adjust the font size as needed
# Remove the color bar generation code for the last row
plt.savefig('./text_pics/' + 'lowtime' + '.png')


# %%


# %%


# %%



