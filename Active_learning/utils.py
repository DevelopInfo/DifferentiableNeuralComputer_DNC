# -*- coding: utf-8 -*-

import json
import linecache
import numpy as np
import codecs

ENCODED_TRAINING_FILE = "encoded_train.json"
TRAIN_FILE = "train.json"
WORD_DICTIONARY = "word_dict.txt"
DICT_FILE = "word_dict.txt"
STOP_WORD = "stop_word.txt"


def read_batch_data(start_line,
                    batch_size,
                    encoded_train_file=ENCODED_TRAINING_FILE):
    """Read one batch size data started from start_line.
    Args:
        start_line: the index for start line,
            which may be larger than file length
        batch_size: the number for read data size
        encoded_train_file: require a json format file
    Returns:
        input_list: a ndarray whose shape is [batch_size, time]
            and dtype is np.float32
        label_list: a ndarray whose shape is [batch_size] and
            dtype is np.float32
    """
    input_list = []
    label_list = []
    end = len(open(encoded_train_file, 'r').readlines())
    counter = batch_size
    linenum = start_line % end + 1

    while counter > 0:
        if linenum > end:
            linenum = 1
        json_str = linecache.getline(filename=encoded_train_file, lineno=linenum)
        linenum += 1
        counter -= 1
        json_dict = json.loads(json_str)
        input_list.append(json_dict['input'])
        label_list.append(json_dict['label'])

    # Convert string to number
    for i in range(len(input_list)):
        input_list[i] = input_list[i].split()
        for index, word in enumerate(input_list[i]):
            input_list[i][index] = int(word)
    input_array = np.array(input_list, dtype=np.int64)

    for i in range(len(label_list)):
        label_list[i] = int(label_list[i])
    label_array = np.array(label_list, dtype=np.int64)

    return input_array, label_array


def decode_and_write_data(input,
                          word_dictionary=WORD_DICTIONARY,
                          decoded_training_file=TRAIN_FILE):
    """Decode data and write data to decoded_train_file.
    Args:
        input: a tensor whose shape is [batch_size, time], dtype is int64
        word_dictionary: a dictionary of word
        decoded_training_file: decoded training file
    """
    # Write json file
    with codecs.open(decoded_training_file, "a", "utf-8") as training_obj:
        for batch_index in range(input.shape[0]):
            sentence = ""
            for word_index in range(input.shape[1]):
                word = linecache.getline(filename=word_dictionary,
                                         lineno=input[batch_index][word_index])
                word = word[0:-1]
                sentence += word
            json_dict = {"label": "", "input": sentence}
            json_str = json.dumps(json_dict).encode('utf-8').decode('unicode-escape')
            training_obj.write(json_str+"\n")


def encode_training_data(train_file=TRAIN_FILE,
                         dict_file=DICT_FILE,
                         stop_file=STOP_WORD,
                         encoded_training_file=ENCODED_TRAINING_FILE):
    """Encoding training data
    Args:
        train_file: require to a json format file.
        encoded_training_file: a encoded training file is also a json format file.
    """
    # Create a dictionary.
    dict = {}
    dict_counter = 1
    with open(dict_file, 'r', encoding="utf-8") as dict_obj:
        for line in dict_obj:
            for word in line:
                if word == "\n":
                    continue
                dict[word] = dict_counter
                dict_counter += 1

    # Create a list of stop word
    stop_word_list = []
    with open(stop_file, 'r', encoding="utf-8") as stop_obj:
        for line in stop_obj:
            for word in line:
                stop_word_list.append(word)

    # Open the encoded training file
    encoded_training_obj = open(encoded_training_file, 'w',
                                encoding="utf-8")
    # Read json file:
    with open(train_file, 'r', encoding="utf-8") as file_obj:
        for line in file_obj:
            decoded_dict = json.loads(line)
            encoded_dict = {}
            encoded_dict["label"] = decoded_dict["label"]
            encoded_dict["input"] = ""
            for word in decoded_dict["input"]:
                if word in stop_word_list:
                    continue
                encoded_dict["input"] = encoded_dict["input"] + "%d" % dict[word]+" "
            encoded_str = json.dumps(encoded_dict)
            encoded_training_obj.write(encoded_str+"\n")
    encoded_training_obj.close()


if __name__ == "__main__":
    """Set parameters."""
    batch_size = 10
    time = 12

    """Create data."""
    input = np.random.randint(
        low=1, high=547, size=(batch_size, time)
    )
    # print(type(input))

    # Test
    # decode_and_write_data(input)

    # encode_training_data()

    read_batch_data(start_line=18, batch_size=batch_size)
