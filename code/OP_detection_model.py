# %%
import numpy as np
import mne
import json
import scipy.stats
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import random

from sklearn.model_selection import train_test_split, GroupShuffleSplit, LeaveOneGroupOut, KFold, GroupKFold, cross_val_score, cross_val_predict, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier
from scipy.stats import ttest_ind


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
picked_channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2", ]
total_channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2", ]
fake_info = mne.create_info(ch_names=total_channels, sfreq=1000., ch_types='eeg')
select_index = [idx for idx in range(len(total_channels)) if total_channels[idx] in picked_channels]

# %%
def compare_strings(A, B):
    # Check if the lengths of the two strings are equal; if not, directly return 0
    if len(A) != len(B):
        return 0

    # Initialize the counter for different characters
    diff_count = 0

    # Traverse each character in the string
    for a, b in zip(A, B):
        # If the characters are different
        if a != b:
            # Check if the different characters are the digits 1 and 2
            if (a, b) not in [('1', '2'), ('2', '1')]:
                return 0
            # Increment the count of different characters
            diff_count += 1
            # If there is more than one different character, directly return 0
            if diff_count > 1:
                return 0

    # If there is only one different digit character and that digit is 1 and 2, return 1; otherwise, return 0
    return 1 if diff_count == 1 else 0


# %%

def get_every_results(behavior_flag, affective_flag, question_flag, select_flag, eeg_flag, like_flag=1, time_flag=1):
    # Function to run various feature combinations
    dates=['fanhao', 'dongyimeng', 'houlinzhi', 'jiwenjun',  'miaoshengze','wanfangwei','wangxiaoting', 'wangzhengni', 'yangchen', 'huqifan',
           'zhangxue','liangqihang', 'daisiwei', 'zhangyutong', 'mengfanjie', 'zhangchenxi', 'liangyanshu','cangyueyang', 'hongyurui','lujianing',
           'zhaochensong','chenrong','chenxingyu']
    dates_accuracy={}
    dates_auc={}
    dates_f1={}
    feature_name="hope_features"
    selected_band=3
    # Initialize global variables
    X = []  # Store EEG features of all subjects
    Y = []  # Store the corresponding FIELD study tendency of all subjects
    eeg_data=[]
    log_data=[]
    behavior_data=[]
    groups = []  # Store the subject identifier for each sample
    def check_v_in_v2info(v, v2info_LAB):
        try:
            value = v2info_LAB[v]
        except KeyError:
            # If v1 is not in v2info_LAB1, a KeyError exception is caught, and the function returns 1
            return 1
        # If no exception is raised, it means v1 is in v2info_LAB1, and the function can return 0 or the value
        return 0
    for date in dates:
        v2info_LAB1 = json.load(open('./v2info/'+'LAB1-'+date+'_v2info.json'))
        idx2de_LAB1 = json.load(open('./'+feature_name+'/'+'LAB1-'+date+'_idx2de.json'))
        v2info_LAB2 = json.load(open('./v2info/'+'LAB2-'+date+'_v2info.json'))
        idx2de_LAB2 = json.load(open('./'+feature_name+'/'+'LAB2-'+date+'_idx2de.json'))
        for v1 in v2info_LAB1.keys():
            for v2 in v2info_LAB2.keys():
                #if(check_string(v1)==0):
                    #continue
                if(v2info_LAB1[v1]['video_type'] >0):
                    continue
                if(v2info_LAB2[v2]['video_type'] >0):
                    continue
                v1_fu=v1.replace("正面","负面")
                v2_fu=v2.replace("正面","负面")
                if(check_v_in_v2info(v1_fu,v2info_LAB1) or check_v_in_v2info(v2_fu,v2info_LAB2)):
                    continue
                if 'idx' not in v2info_LAB1[v1].keys() or v2info_LAB1[v1]['tend'] == 2 or 'idx' not in v2info_LAB1[v1_fu].keys():
                    continue
                if 'idx' not in v2info_LAB2[v2].keys() or v2info_LAB2[v2]['tend'] == 2 or 'idx' not in v2info_LAB2[v2_fu].keys():
                    continue
                if(compare_strings(v1,v2)):
                    #print(date,v1)
                    feature_1 = np.array(idx2de_LAB1[str(v2info_LAB1[v1]['idx'])])
                    feature_2 = np.array(idx2de_LAB2[str(v2info_LAB2[v2]['idx'])])
                    feature_1_negative = np.array(idx2de_LAB1[str(v2info_LAB1[v1_fu]['idx'])])
                    feature_2_negative = np.array(idx2de_LAB2[str(v2info_LAB2[v2_fu]['idx'])])
                    feature_1_va= v2info_LAB1[v1]['valence']
                    feature_2_va= v2info_LAB2[v2]['valence']
                    feature_1_ro= v2info_LAB1[v1]['arousal']
                    feature_2_ro= v2info_LAB2[v2]['arousal']
                    feature_1_time= v2info_LAB1[v1]['play_duration']
                    feature_2_time= v2info_LAB2[v2]['play_duration']
                    feature_1_like= v2info_LAB1[v1]['like']
                    feature_2_like= v2info_LAB2[v2]['like']
                    feature_1_question = v2info_LAB1[v1]['question']
                    feature_2_question = v2info_LAB2[v2]['question']
                    feature_1_familarity = v2info_LAB1[v1]['familarity']
                    feature_2_familarity = v2info_LAB2[v2]['familarity']
                    feature_1_va_negative=v2info_LAB1[v1_fu]['valence']
                    feature_2_va_negative=v2info_LAB2[v2_fu]['valence']
                    feature_1_ro_negative= v2info_LAB1[v1_fu]['arousal']
                    feature_2_ro_negative= v2info_LAB2[v2_fu]['arousal']
                    feature_1_time_negative= v2info_LAB1[v1_fu]['play_duration']
                    feature_2_time_negative= v2info_LAB2[v2_fu]['play_duration']
                    feature_1_like_negative= v2info_LAB1[v1_fu]['like']
                    feature_2_like_negative= v2info_LAB2[v2_fu]['like']
                    log_data_list=[]
                    behavior_list=[]
                    if(behavior_flag==1):
                        like_flag=1
                        time_flag=0
                    if(like_flag==1):
                        #log_data_list.append(feature_1_like)
                        #log_data_list.append(feature_2_like)
                        #log_data_list.append(feature_1_like_negative)
                        #log_data_list.append(feature_2_like_negative)
                        behavior_list.append(feature_1_like)
                        behavior_list.append(feature_2_like)
                        behavior_list.append(feature_1_like_negative)
                        behavior_list.append(feature_2_like_negative)
                        #log_data_list.append(feature_2_like-feature_1_like)
                        #log_data_list.append(feature_2_like_negative-feature_1_like_negative)
                    if(time_flag==1):
                        #log_data_list.append(feature_1_time)
                        #log_data_list.append(feature_2_time)
                        #log_data_list.append(feature_1_time_negative)
                        #log_data_list.append(feature_2_time_negative)
                        behavior_list.append(feature_1_time)
                        behavior_list.append(feature_2_time)
                        behavior_list.append(feature_1_time_negative)
                        behavior_list.append(feature_2_time_negative)
                        #log_data_list.append(feature_2_time-feature_1_time)
                        #log_data_list.append(feature_2_time_negative-feature_1_time_negative)
                    if(affective_flag==1):
                        log_data_list.append(feature_1_va)
                        log_data_list.append(feature_2_va)
                        log_data_list.append(feature_1_va_negative)
                        log_data_list.append(feature_2_va_negative)
                    if(question_flag==1):
                        #log_data_list.append(feature_1_question)
                        #log_data_list.append(feature_1_question)
                        log_data_list.append(feature_2_question)
                        #log_data_list.append(feature_2_question)
                        #log_data_list.append(feature_1_familarity)
                        log_data_list.append(feature_2_familarity)
                    # Testing random results
                    random_number = random.randint(0, 1)
                    #for i in range(feature_1.shape[0]):
                    if(len(feature_1)==len(feature_2_negative)):
                    #for i in range(1):
                        if(feature_2.shape != feature_1.shape):
                            break
                        if(feature_2_negative.shape != feature_1_negative.shape):
                            break
                        #feature = np.concatenate((feature_1,feature_2,feature_1_negative,feature_2_negative), axis=0)
                        feature = np.stack((feature_1, feature_2, feature_1_negative, feature_2_negative), axis=0)
                        #print(feature.shape)
                        # Merge the weighted features
                        #feature = np.concatenate((weighted_feature1, weighted_feature2), axis=1)
                        #print(feature[i].shape)
                        #flattened_feature = feature.flatten()
                        ave_feature=np.mean(feature,axis=1)
                        flattened_feature = ave_feature.flatten()
                        #print(ave_feature.shape)
                        #print(flattened_feature.shape)
                        #log_info=np.array([feature_1_va,feature_2_va,feature_1_ro,feature_2_ro,feature_1_like,feature_2_like,feature_1_time,feature_2_time])
                        if not X or len(flattened_feature) == len(X[0]):
                            X.append(flattened_feature)
                            #log_data.append(np.array([feature_1_ro,feature_2_ro,feature_1_ro_negative,feature_2_ro_negative]))
                            #log_data.append(np.array([feature_1_question,feature_2_question,feature_1_va,feature_2_va,feature_1_va_negative,feature_2_va_negative,feature_1_time,feature_2_time,feature_1_time_negative,feature_2_time_negative,feature_1_like,feature_2_like,feature_1_like_negative,feature_2_like_negative]))
                            #log_data.append(np.array([feature_1_time,feature_2_time,feature_1_time_negative,feature_2_time_negative]))
                            #log_data.append(np.array([feature_1_time_negative,feature_2_time_negative]))
                            #log_data.append(np.array([feature_1_like,feature_2_like,feature_1_like_negative,feature_2_like_negative]))
                            #log_data.append(np.array([feature_1_question,feature_2_question,feature_1_va,feature_2_va,feature_1_va_negative,feature_2_va_negative]))
                            log_data.append(np.array(log_data_list))
                            eeg_data.append(flattened_feature)
                            behavior_data.append(behavior_list)
                            Y.append(v2info_LAB2[v2]['tend'])
                            #Y.append(random_number)
                            groups.append(date)  # Add subject identifier for each sample
                        else:
                            print(f"Length mismatch in feature for index {v1}: Expected {len(X[0])}, got {len(flattened_feature)}")
    # Convert to NumPy array
    X = np.array(X)
        #print(X.shape)
    # Feature scaling
    scaler = StandardScaler()
    eeg_scaled = scaler.fit_transform(eeg_data)
    #eeg_selected = eeg_scaled[:, [i * 5 + j for j in range(1,4) for i in range(5)]]
    #eeg_selected = eeg_scaled[:, [i * 5 + j for j in range(1,3) for i in range(5)]]
    #eeg_selected = eeg_scaled[:, [i * 5 + j for j in range(1,3) for i in [3,4,7,11]]]
    #print(eeg_scaled.shape)
    #eeg_selected = eeg_scaled[:, [s*9300+t*310+i * 5 + j for j in range(1,3) for i in [3,4] for s in [0,1,2,3] for t in [0,1,2]]]
    #print(eeg_selected.shape)
    eeg_selected = eeg_scaled[:, [s*310+i * 5 + j for j in [2] for i in [1] for s in [0,1,2,3]]]
    # Adjust the number of components for PCA dimension reduction
    pca = PCA(n_components=1)  # Adjust n_components as needed
    epca = PCA(n_components=2)  # Adjust n_components as needed
    #eeg_pca = epca.fit_transform(eeg_selected)
    #eeg_selected=eeg_pca
    if(behavior_flag==1):
        behavior_pca = pca.fit_transform(behavior_data)
    # Increase the weight of log data
    #log_data_weighted = np.tile(log_data * 2, (eeg_pca.shape[0], 1))  # Adjust the weighting factor
    # Horizontally merge features
        log_data = np.hstack((np.array(log_data), np.array(behavior_pca))).tolist()
    behavior_pca = pca.fit_transform(behavior_data)
    print(np.array(log_data).shape)
    log_data= behavior_data
    #log_data = np.hstack((np.array(log_data), np.array(behavior_pca))).tolist()
    #combined_features = np.array(log_data)
    if(select_flag == 1):
        my_eeg = eeg_selected
    else:
        my_eeg = eeg_scaled
    if(eeg_flag == 0):
        combined_features = np.array(log_data)
    elif(behavior_flag+affective_flag+question_flag > 0):
        combined_features = np.hstack((np.array(log_data)*18600, my_eeg))
    else:
        combined_features = np.array(my_eeg)
    X=combined_features
    
    Y = np.array(Y)
    print(X.shape)
        #print("Y mean value:", Y.mean())
    # Display the number of 0's and 1's in Y
    num_zeros = sum(Y == 0)
    num_ones = sum(Y == 1)
    print(f"Number of 0's: {num_zeros}")
    print(f"Number of 1's: {num_ones}")
    
    # Create and train the classification model
    # Create an SVM classification model
    #classifier = SVC(probability=True)
    # Create a Gradient Boosting classification model
    #classifier = GradientBoostingClassifier(n_estimators=100)
    #classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)
    # Create an XGBoost classification model
    classifier = XGBoostClassifier(use_label_encoder=False, eval_metric='logloss')
    #classifier = RandomForestClassifier()  # You can choose other classifiers
    #classifier = LogisticRegression(random_state=42)
    #classifier = RandomForestClassifier()  # You can choose other classifiers
    # Prepare for five-fold cross-validation and calculate accuracy and AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=81)
    scoring = ['accuracy', 'roc_auc', 'f1']
    cv_results = cross_validate(classifier, X, Y, cv=cv, scoring=scoring, return_estimator=True)
    
    #print("Accuracy scores for each fold:", cv_results['test_accuracy'])
    #print("Average accuracy:", np.mean(cv_results['test_accuracy']))
    #print("AUC scores for each fold:", cv_results['test_roc_auc'])
    #print("Average AUC:", np.mean(cv_results['test_roc_auc']))
    #print("F1 scores for each fold:", cv_results['test_f1'])
    #print("Average F1 score:", np.mean(cv_results['test_f1']))
    my_auc=np.mean(cv_results['test_roc_auc'])
    my_f1=np.mean(cv_results['test_f1'])
    my_acc=np.mean(cv_results['test_accuracy'])

    return my_auc,my_f1,my_acc


# %%
# set flags
all_results={}
select_results={}
for behavior_flag in [0,1]:
    for affective_flag in [0]:
        for question_flag in [0,1]:
            for select_flag in [1]:
                for eeg_flag in [0,1]:
                    for like_flag in [0]:
                        for time_flag in [0]:
                            if(behavior_flag+affective_flag+question_flag+eeg_flag+like_flag+time_flag==0):
                                continue
                            this_result=get_every_results(behavior_flag,affective_flag,question_flag,select_flag,eeg_flag,like_flag,time_flag)
                            print("parameters：",behavior_flag,question_flag,eeg_flag,like_flag,time_flag)
                            print(this_result,'\n')
                            all_results[(behavior_flag,affective_flag,question_flag,select_flag,eeg_flag)]=this_result
                            if(select_flag==1):
                                select_results[(behavior_flag,affective_flag,question_flag,select_flag,eeg_flag)]=this_result


