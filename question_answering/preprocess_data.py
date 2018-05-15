# -*- coding: utf-8 -*-

import os
import numpy as np


def create_dictionary(files_path):
    files_list = os.listdir(files_path)
    lexicons_dict_file = open('./lexicons_dict.txt','w')

    lexicons_dict = {}
    lexicons_counter = 1

    print("Creating Dictionary ... 0/%d" % (len(files_list)))

    for file_index, file_name in enumerate(files_list):
        with open(files_path+file_name, 'r') as file_obj:
            for line in file_obj:
                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')
                for word in line.split():
                    if word.lower() not in lexicons_dict and word.isalpha():
                        lexicons_dict_file.write(word.lower()+"\n")
                        lexicons_dict[word.lower()] = lexicons_counter
                        lexicons_counter += 1
        print("\rCreating Dictionary ... %d/%d" % (file_index, len(files_list)))

    print("\rCreating Dictionary ... Done!")
    lexicons_dict_file.close()
    return lexicons_dict


def encode_data(files_path, lexicons_dictionary):
    files_list = os.listdir(files_path)

    for file_index, file_name in enumerate(files_list):
        story = ''
        story_file = open('./data/story_'+file_name, 'w')
        query_file = open('./data/query_'+file_name, 'w')
        answer_file = open('./data/answer_'+file_name, 'w')
        ans_index_file = open('./data/ans_index_'+file_name, 'w')
        with open(files_path+file_name, 'r') as file_obj:
            query_tag = False
            for line in file_obj:
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')
                line_str = ''
                query_str = ''
                ans_str = ''
                ans_index_str = ''
                for word_index, word in enumerate(line.split()):
                    if word.isalpha():
                        line_str += '%d ' % lexicons_dictionary[word.lower()]
                    # the tag to begin story
                    if word_index == 0 and word == '1':
                        story = ''
                    if word == '?':
                        query_tag = True
                        query_str += line_str
                    if query_tag and word.isalpha():
                        ans_str += '%d ' % lexicons_dictionary[word.lower()]
                    if query_tag and word.isdigit():
                        ans_index_str += '%s ' % word
                if not query_tag:
                    story += line_str
                else:
                    # print(file_name)
                    # print('story_str: \n' + story)
                    # print('query_str: \n' + query_str)
                    # print('ans_str: \n' + ans_str)
                    # print('ans_index_str: \n' + ans_index_str)
                    story_file.write(story+'\n')
                    query_file.write(query_str+'\n')
                    answer_file.write(ans_str+'\n')
                    ans_index_file.write(ans_index_str+'\n')
                    query_tag=False

        story_file.close()
        query_file.close()
        answer_file.close()
        ans_index_file.close()


def _batch_pad(data, val=0, front=False):
    """Padding batch data."""
    T = max(map(len, data))

    padded_data = []
    for sent in data:
        N = len(sent)
        if N < T:
            _num = T-N
            # front padding
            if front:
                padded_data.append(
                    np.pad(sent, (_num, 0), 'constant', constant_values=(val, val))
                )
            else:
                padded_data.append(
                    np.pad(sent, (0, _num), 'constant', constant_values=(val, val))
                )
        else:
            padded_data.append(
                np.pad(sent, (0, 0), 'constant', constant_values=(val, val))
            )

    return padded_data


def _data_pad(file_path):
    with open(file_path, 'r') as file_read:
        data = []
        for line in file_read:
            # remove '\n'
            line = line[:-1]
            # remove the last element
            line = line.strip()
            # split the word embedding
            line = line.split(' ')
            # print(line)
            data_line = []
            for word in line:
                data_line.append(int(word))
            data.append(data_line)
        padded_data = _batch_pad(data)
        # print(padded_data)
        return padded_data


import tensorflow as tf


def make_train_tfrecord(file_name,
                        story_file,
                        query_file,
                        ans_file,
                        ans_index_file):
    # define a tfrecord writer
    writer = tf.python_io.TFRecordWriter(file_name+"_train.tfrecords")

    story_value = _data_pad(story_file)
    query_value = _data_pad(query_file)
    ans_value = _data_pad(ans_file)
    ans_index_value = _data_pad(ans_index_file)

    for i in range(len(story_value)-1):
        print(len(story_value[i]))
        print(len(query_value[i]))
        print(len(ans_value[i]))
        print(len(ans_index_value[i]))
        story_i = story_value[i].tostring()
        query_i = query_value[i].tostring()
        ans_i = ans_value[i].tostring()
        ans_index_i = ans_index_value[i].tostring()
        # define and initial a map<string, Feature>
        feature = {
            "story": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[story_i])),
            "query": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[query_i])),
            "ans": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[ans_i])),
            "ans_index": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[ans_index_i]))
        }
        # define and initial a example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # print(example)

        # write the example protocol buffer to the file
        writer.write(example.SerializeToString())

    writer.close()

# lexicons_dict = create_dictionary("./tasks_1-20_v1-2/en/")
# encode_data("./tasks_1-20_v1-2/en/", lexicons_dict)
#_data_pad("./data/story_qa1_single-supporting-fact_train.txt")
make_train_tfrecord(file_name="qa1",
                    story_file="./data/story_qa1_single-supporting-fact_train.txt",
                    query_file="./data/query_qa1_single-supporting-fact_train.txt",
                    ans_file="./data/answer_qa1_single-supporting-fact_train.txt",
                    ans_index_file="./data/ans_index_qa1_single-supporting-fact_train.txt")