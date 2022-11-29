# coding=utf-8
from __future__ import unicode_literals
import gc
import io
import json
from model.trend import DIAL
# from model.model_tgn_noContent import DIAL
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.model_selection import train_test_split
from nltk import tokenize
from keras.utils.np_utils import to_categorical
import operator
import codecs
import os
from keras import backend as K
import argparse
import random
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

# when gpu>1 choose a gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DATA_PATH = 'data'
SAVED_MODEL_DIR = 'saved_models'
SAVED_MODEL_FILENAME = 'DIAL_model.h5'
EMBEDDINGS_PATH = 'saved_models/glove.6B.100d.txt'


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip()


if __name__ == '__main__':
    platform = 'weibo_1'
    parser = argparse.ArgumentParser(description="your script description")
    parser.add_argument('--sent_len', '-sl', help='sentence_length')
    parser.add_argument('--sent_cout', '-sc', help='sentence_count')
    parser.add_argument('--comt_len', '-cl', help='comment_length')
    parser.add_argument('--comt_cout', '-cc', help='comment_count')
    parser.add_argument('--user_comt', '-uc', help='user_comment')
    parser.add_argument('--platform', '-pf', help='platform')
    parser.add_argument('--learning', '-lr', help='learning_rate')
    parser.add_argument('--bat_size', '-bs', help='batch_size')
    parser.add_argument('--explain_file_name', '-efn', help='explain_file_name')
    args = parser.parse_args()

    if args.platform:
        platform = args.platform

    data_train = pd.read_csv('data/' + platform + '/' + platform + '_content_no_ignore_2.tsv', sep='\t', encoding="utf-8")
    contents_vector = io.open('data/' + platform + '/' + platform + "_content_vector_200_50_1_2.txt", "r", encoding="utf-8")
    VALIDATION_SPLIT = 0.25
    contents = []
    labels = []
    texts = []
    ids = []
    content_vec_dic = {}
    for i in contents_vector:
        i = i.strip("\n")
        e = i.split("$:$:")
        v = e[1].split(" ")
        for j in range(len(v)):
            v[j] = v[j].lstrip(":")
            v[j] = float(v[j])
        temp = np.empty((1, 200))
        for i in range(200):
            temp[0][i] = v[i]
        v = temp
        content_vec_dic[e[0]] = v

    for idx in range(data_train.content.shape[0]):
        text = data_train.content[idx]
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        tmp_contents_vector = np.empty((0, 200))
        sentences_count = 0
        if len(sentences) > 50:
            # continue
            for s in sentences[:50]:
                tmp_contents_vector = np.append(tmp_contents_vector, content_vec_dic[s], axis=0)
        else:
            for s in sentences:
                # print(s)
                sentences_count += 1
                tmp_contents_vector = np.append(tmp_contents_vector, content_vec_dic[s], axis=0)
            while sentences_count < 50:
                tmp_contents_vector = np.append(tmp_contents_vector, np.zeros((1, 200)), axis=0)  # axis=0，表示针对第1维进行操作，可以简单的理解为，加在了行上。
                sentences_count += 1
        contents.append(tmp_contents_vector)
        ids.append(data_train.id[idx])

        labels.append(data_train.label[idx])
    labels = np.asarray(labels)
    labels = to_categorical(labels)
    del content_vec_dic
    print("content finished!")

    u_id_index_dict = {}
    f_u_id_index = io.open('data/tgn/' + platform + "_user_id.txt", "r", encoding="utf-8")
    for line in f_u_id_index:
        line = line.rstrip('\n')
        e = line.split(' ')
        u_id_index_dict[str(e[0])] = e[1]
    u_vector_dict = {}
    f_u_vector = io.open('data/tgn/' + platform + "_user_vector.txt", "r", encoding="utf-8")
    a = True
    for line in f_u_vector:
        line = line.rstrip('\n')
        e = line.split('$:$:')
        u_real_id = u_id_index_dict[str(e[0])]
        v = e[1].split(" ")
        for j in range(len(v)):
            v[j] = v[j].lstrip(":")
            v[j] = float(v[j])
        temp = np.empty((1, 200))
        for i in range(200):
            temp[0][i] = v[i]
        v = temp
        u_vector_dict[str(u_real_id)] = v

    count_uid = 0
    count = 0
    u_embedding = []
    comment_u_id = pd.read_csv('data/' + platform + '/' + platform + '_comment_id_no_ignore.tsv', sep='\t', encoding="utf-8")
    for idx in range(comment_u_id.comment.shape[0]):
        tmp_u_vector = np.empty((0, 200))
        comments = comment_u_id.comment[idx].split("$:$:")
        if len(comments) > 50:
            comments = comments[:50]
        for u_id in comments:
            count+=1
            temp = np.empty((1, 200))
            for j in range(200):
                if str(u_id) in u_vector_dict:
                    count_uid+=1
                    temp[0][j] = u_vector_dict[str(u_id)][0][j]
                else:
                    temp[0][j] = 0
            tmp_u_vector = np.append(tmp_u_vector, temp, axis=0)
        u_embedding.append(tmp_u_vector)
    print(len(u_embedding),":user finished!")
    print(len(u_embedding[290]))
    print("count_uid_time:",count_uid/200)
    print("count_uid_time:",count)

    u_structure = []
    # u_structure_vector = np.load('data/'+platform+'/'+platform+'_user_node_emd.npy')
    u_structure_vector = np.load('data/'+platform+'/splitBIgru_'+platform+'.npy')
    comment_u_id = pd.read_csv('data/' + platform + '/' + platform + '_comment_id_no_ignore.tsv', sep='\t',
                             encoding="utf-8")
    f_id_index = io.open("data/"+ platform + '/'+platform+"_index_uid.txt",'r', encoding='utf-8')
    # f_user_id = io.open("data/"+platform+'/'+platform+'_users_id.txt', 'r', encoding='utf-8')
    # user_id_dict = {}
    # for line in f_user_id:
    #     line = line.rstrip('\n')
    #     e = line.split(' ')
    #     user_id_dict[int(e[1])] = int(e[0])

    f_user_list = io.open("data/"+platform+'/'+platform+'_user_id_relation_nodelist.txt', 'r', encoding='utf-8')
    user_id_list = []
    for line in f_user_list:
        line = line.rstrip('\n')
        user_id_list.append(line)

    user_id_index_dict = {}
    for line in f_id_index:
        line = line.rstrip('\n')
        e = line.split(" ")
        user_id_index_dict[str(e[0])] = str(e[1])  # 索引值：real_id
    
    user_id_dict = {}
    for i in range(len(user_id_list)):
        temp = np.empty((1, 200))
        for j in range(200):
            temp[0][j] = u_structure_vector[i][j]
        user_id_dict[user_id_index_dict[str(user_id_list[i])]] = temp

    count_uid = 0
    u_structure = []
    for idx in range(comment_u_id.comment.shape[0]):
        tmp_us_vector = np.empty((0, 200))
        comments = comment_u_id.comment[idx].split("$:$:")
        if len(comments) > 50:
            comments = comments[:50]
        for u_id in comments:
            temp = np.empty((1, 200))
            if str(u_id) in user_id_dict:
                count_uid+=1
                temp = user_id_dict[str(u_id)]
            tmp_us_vector = np.append(tmp_us_vector, temp, axis=0)
        u_structure.append(tmp_us_vector)
    print(len(u_structure),":user structure finished!")
    print(len(u_structure[290]))
    print("count_uid:",count_uid)

    # u_structure = []
    # for idx in range(comment_u_id.comment.shape[0]):
    #     tmp_us_vector = np.empty((0, 200))
    #     comments = comment_u_id.comment[idx].split("::")
    #     if len(comments) > 50:
    #         comments = comments[:50]
    #     for u_id in comments:
    #         temp = np.empty((1, 200))
    #         for j in range(200):
    #             if int(u_id) in user_id_dict:
    #                 index = user_id_dict[int(u_id)]
    #                 temp[0][j] = u_structure_vector[index-1][j]
    #             else:
    #                 temp[0][j] = 0
    #         tmp_us_vector = np.append(tmp_us_vector, temp, axis=0)
    #     u_structure.append(tmp_us_vector)
    # print(len(u_structure),":user structure finished!")
    # print(len(u_structure[290]))

    # comment_info = []
    # comment_train = open('data/' + platform + '/' + platform + "_contentID_comment_vector_200_50_all_splitter.txt", "r")
    # article_id = 0
    # tmp_uc_vector = np.empty((0, 200))
    # for line in comment_train:
    #     e = line.split("$:$:")
    #     if int(e[0]) != article_id:
    #         article_id = article_id + 1
    #         comment_info.append(tmp_uc_vector)
    #         tmp_uc_vector = np.empty((0, 200))
    #     v = e[2].split(" ")
    #     for j in range(len(v)):
    #         v[j] = v[j].lstrip(":")
    #         v[j] = float(v[j])
    #     temp = np.empty((1, 200))
    #     for i in range(200):
    #         # temp[0][i] = v[i]
    #         temp[0][i] = 0
    #     v = temp
    #     tmp_uc_vector = np.append(tmp_uc_vector, v, axis=0)
    # comment_info.append(tmp_uc_vector)
    # print(len(comment_info), ":comment finished!")
    # print(len(comment_info[290]))

    id_train, id_test, x_train, x_val, x_train_u, x_val_u, x_train_us,x_val_us,y_train, y_val = train_test_split(
        ids,
        contents,
        u_embedding,
        u_structure,
        labels,
        # comment_info,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=labels)
    # for split data save
    print(id_test)
    '''
    with codecs.open('./x_train.txt', 'w', encoding='utf-8') as f:
        for i in x_train:
            f.write(":::".join(i) + "\n")
    np.savetxt('./y_train.txt', y_train, fmt=str('%f'), delimiter=',', encoding='utf-8')
    with codecs.open('./c_train.txt', 'w', encoding='utf-8') as f:
        for i in c_train:
            f.write(":::".join(i) + "\n")
    with codecs.open('./cid_train.txt', 'w', encoding='utf-8') as f:
        for i in cid_train:
            f.write(":::".join(i) + "\n")
    with codecs.open('./cid_val.txt', 'w', encoding='utf-8') as f:
        for i in cid_val:
            f.write(":::".join(i) + "\n")
    with codecs.open('./c_val.txt', 'w', encoding='utf-8') as f:
        for i in c_val:
            f.write(":::".join(i) + "\n")
    with codecs.open('./x_val.txt', 'w', encoding='utf-8') as f:
        for i in x_val:
            f.write(":::".join(i) + "\n")
    np.savetxt("./y_val.txt", y_val, fmt=str('%f'), delimiter=',', encoding='utf-8')
    '''

    # load train and valid data
    '''
    x_train = []
    with codecs.open('./x_train_' + platform + '.txt', 'r', encoding='utf-8') as f:
        line = f.readline().strip('\n')
        while line:
            ls = line.split(':::')
            x_train.append(ls)
            line = f.readline().strip('\n')
    x_val = []
    with codecs.open('./x_val_' + platform + '.txt', 'r', encoding='utf-8') as f:
        line = f.readline().strip('\n')
        while line:
            ls = line.split(':::')
            x_val.append(ls)
            line = f.readline().strip('\n')
    c_train = []
    with codecs.open('./c_train_' + platform + '.txt', 'r', encoding='utf-8') as f:
        line = f.readline().strip('\n')
        while line:
            ls = line.split(':::')
            c_train.append(ls)
            line = f.readline().strip('\n')
    c_val = []
    with codecs.open('./c_val_' + platform + '.txt', 'r', encoding='utf-8') as f:
        line = f.readline().strip('\n')
        while line:
            ls = line.split(':::')
            c_val.append(ls)
            line = f.readline().strip('\n')
    cid_train = []
    with codecs.open('./cid_train_' + platform + '.txt', 'r', encoding='utf-8') as f:
        line = f.readline().strip('\n')
        while line:
            ls = line.split(':::')
            cid_train.append(ls)
            line = f.readline().strip('\n')
    cid_val = []
    with codecs.open('./cid_val_' + platform + '.txt', 'r', encoding='utf-8') as f:
        line = f.readline().strip('\n')
        while line:
            ls = line.split(':::')
            cid_val.append(ls)
            line = f.readline().strip('\n')
    y_train = np.loadtxt('./y_train_' + platform + '.txt',dtype = np.float,delimiter=',')
    y_val = np.loadtxt('./y_val_' + platform + '.txt',dtype = np.float,delimiter=',')
    '''

    # Train and save the GCANmodel
    # batch_size = 20
    # SAVED_MODEL_FILENAME = platform + '_DIAL_new_model.h5'
    # if args.sent_len:
    #     h = DIAL(platform, MAX_SENTENCE_LENGTH=int(args.sent_len), alter_params='sent_len')
    # elif args.comt_len:
    #     h = DIAL(platform, MAX_COMS_LENGTH=int(args.comt_len), alter_params='comt_len')
    # elif args.comt_cout:
    #     h = DIAL(platform, MAX_COMS_COUNT=int(args.comt_cout), alter_params='comt_cout')
    # elif args.sent_cout:
    #     h = DIAL(platform, MAX_SENTENCE_COUNT=int(args.sent_cout), alter_params='sent_cout')
    # elif args.user_comt:
    #     h = DIAL(platform, USER_COMS=int(args.user_comt), alter_params='user_comt')
    # elif args.learning:
    #     print(round(float(args.learning) / 10000, 6))
    #     h = DIAL(platform, lr=round(float(args.learning) / 10000, 6), alter_params='lr')
    # elif args.bat_size:
    #     h = DIAL(platform, alter_params='batch_size')
    #     batch_size = int(args.bat_size)
    # else:
    #     h = DIAL(platform, alter_params='null')
    
    h = DIAL(platform, lr=round(float(args.learning) / 10000, 6), alter_params='bs_lr')
    batch_size = int(args.bat_size)
    print(platform)
    print("lr:",round(float(args.learning) / 10000, 6))
    print("batch_size:",batch_size)

    h.train(x_train, x_train_u,x_train_us, y_train,x_val, x_val_u,x_val_us, y_val,
            batch_size=batch_size,
            epochs=50,
            embeddings_path='./glove.6B.100d.txt',
            saved_model_dir=str(SAVED_MODEL_DIR),
            saved_model_filename=str(SAVED_MODEL_FILENAME))

    h.load_weights(saved_model_dir=str(SAVED_MODEL_DIR), saved_model_filename=str(SAVED_MODEL_FILENAME))
    # for explain

    # result = h.predict(x_val,c_val,cid_val)
    # print(result)
    # # Get the attention weights for sentences in the news contents as well as comments
    # activation_maps = h.activation_maps(x_val, c_val, cid_val)

    # #######
    # activation_maps = h.activation_maps(x_val, c_val, cid_val)
    # with codecs.open('./explain/'+str(platform)+'_results_'+str(args.explain_file_name)+'.txt', 'w', encoding='utf-8') as f:
    #     for i in range(len(activation_maps[0])):
    #             news_s_attention = ""
    #             for j in range(len(activation_maps[0][i])-1):
    #                 news_s_attention += str(round(float(activation_maps[0][i][j]),6)) + "::"
    #             news_s_attention += str(activation_maps[0][i][len(activation_maps[0][i])-1])
    #             f.write(str(id_test[i]) + '\t' + news_s_attention + '\n')
    # ######

    # for i in range(len(activation_maps[1])):
    #     if len(activation_maps[1][i]) >= 30:
    #         with codecs.open('./explain/{}_result_comment_{}.txt'.format(platform, str(i)), 'w', encoding='utf-8') as f:
    #             for j in range(len(activation_maps[1][i])):
    #                 f.write(" ".join(activation_maps[1][i][j][0]) + '\t' + str(activation_maps[1][i][j][1]) + '\n')
