# -*- coding: utf-8 -*-

import json
import linecache
import numpy as np
import codecs

TEST_PATH = "text.txt"

TRAIN_FILE = "train.json"
TEST_FILE = "test.json"

WORD_DICTIONARY = "word_dict.txt"
STOP_WORD = "stop_word.txt"


def encoding_and_read_data(raw_file=TRAIN_FILE,
                  dict_file=WORD_DICTIONARY,
                  stop_file=STOP_WORD):
    """Encoding and read data
    Args:
        raw_file: require to a json format file.
    Returns:
        label_and_input: a list which contains all samples
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

    label_and_input = []
    # Read json file:
    with open(raw_file, 'r', encoding="utf-8") as file_obj:
        for line in file_obj:
            decoded_dict = json.loads(line)

            single_sample_dict = {}
            # Process label.
            single_sample_dict["label"] = []
            if decoded_dict['label'].isdigit():
                single_sample_dict["label"].append(int(decoded_dict['label']))
            else:
                single_sample_dict["label"].append(0)

            # Process input.
            single_sample_dict["input"] = []
            for word in decoded_dict["input"]:
                if word != '0' and word in stop_word_list:
                    continue
                if word == '0':
                    single_sample_dict["input"].append(0)
                else:
                    single_sample_dict["input"].append(dict[word])

            # Append to label_and_input
            label_and_input.append(single_sample_dict)

    return label_and_input


def _batch_pad(data, val=0, front=True):
    """
    Padding batch data.

    Arguments:
        data {[list]} -- [description]

    Keyword Arguments:
        val {[object]} -- [description] (default: {0})
        front {[bool]} -- [description] (default: {True})

    Returns:
        [type] -- [description]
    """
    T = max(map(len, data))

    padded_data = []
    for sent in data:
        N = len(sent)
        if N < T:
            _num = T - N
            # front padding
            if front:
                padded_data.append(
                    np.pad(sent, (_num, 0), 'constant', constant_values=(val))
                )
            else:
                padded_data.append(
                    np.pad(sent, (0, _num), 'constant', constant_values=(val))
                )
        else:
            padded_data.append(np.array(sent))

    return padded_data

def read_batch_data(pos, batch_size, data_set):
    """Read one batch data.
    Args:
        pos: the start pos
        batch_size: batch size for reading
        data_set: a list contains all sample
    Returns:
        batch_label: an array whose shape is [batch_size, ]
            and dtype is int64
        batch_label: an array whose shape is [batch_size, time]
            and dtype is int64
        index: represents the end data index
    """
    batch_label = []
    batch_input = []
    for index in range(batch_size):
        try:
            batch_label.append(data_set[pos + index]["label"])
            batch_input.append(data_set[pos + index]["input"])
        except IndexError:
            if batch_input != []:
                # Padding data.
                batch_input = _batch_pad(batch_input)
            batch_label = np.array(batch_label, dtype=np.int64)
            batch_input = np.array(batch_input, dtype=np.int64)
            return batch_label, batch_input, index

    # Padding data.
    batch_input = _batch_pad(batch_input)
    batch_label = np.array(batch_label, dtype=np.int64)
    batch_input = np.array(batch_input, dtype=np.int64)
    return batch_label, batch_input, batch_size


def decode_and_write_data(input,
                          word_dictionary=WORD_DICTIONARY,
                          decoded_training_file=TRAIN_FILE):
    """Decode data and write data to decoded_train_file.
    Args:
        input: a list which contains input sequence and dtype is int64
        word_dictionary: a dictionary of word
        decoded_training_file: decoded training file
    """
    # Write json file
    with codecs.open(decoded_training_file, "a", "utf-8") as training_obj:
        for batch_index in range(len(input)):
            sentence = ""
            for word_index in range(len(input[batch_index])):
                if input[batch_index][word_index] == 0:
                    sentence += '0 '
                else:
                    word = linecache.getline(filename=word_dictionary,
                                             lineno=input[batch_index][word_index])
                    word = word[0:-1]
                    sentence += word
            json_dict = {"label": "", "input": sentence}
            json_str = json.dumps(json_dict).encode('utf-8').decode('unicode-escape')
            training_obj.write(json_str+"\n")


# ENCODED_TRAINING_FILE = "encoded_train.json"
# ENCODED_TEST_FILE = "encoded_test.json"


# def read_batch_data_v1(start_line,
#                     batch_size,
#                     encoded_file):
#     """Read one batch size data started from start_line.
#     Args:
#         start_line: the index for start line,
#             which may be larger than file length
#         batch_size: the number for read data size
#         encoded_file: require a json format file
#     Returns:
#         input_list: a ndarray whose shape is [batch_size, time]
#             and dtype is np.float32
#         label_list: a ndarray whose shape is [batch_size] and
#             dtype is np.float32
#     """
#     input_list = []
#     label_list = []
#     end = len(open(encoded_file, 'r').readlines())
#     counter = batch_size
#     linenum = start_line % end + 1
#
#
#     while counter > 0:
#         if linenum > end:
#             linenum = 1
#         json_str = linecache.getline(filename=encoded_file, lineno=linenum)
#         linenum += 1
#         counter -= 1
#         json_dict = json.loads(json_str)
#         input_list.append(json_dict['input'])
#         label_list.append(json_dict['label'])
#
#     # Convert string to number
#     for i in range(len(input_list)):
#         input_list[i] = input_list[i].split()
#         for index, word in enumerate(input_list[i]):
#             input_list[i][index] = int(word)
#     input_array = np.array(input_list, dtype=np.int64)
#
#     for i in range(len(label_list)):
#         if label_list[i].isdigit():
#             label_list[i] = int(label_list[i])
#         else:
#             label_list[i] = int('0')
#     label_array = np.array(label_list, dtype=np.int64)
#
#     return input_array, label_array, linenum


if __name__ == "__main__":
    """Set parameters."""
    batch_size = 10
    time = 12

    """Create data."""
    input = np.random.randint(
        low=1, high=547, size=(batch_size, time))
