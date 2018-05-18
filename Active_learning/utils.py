# -*- coding: utf-8 -*-

import json
import linecache
import numpy as np
import codecs

ENCODED_TRAINING_FILE = "encoded_train.json"
TRAIN_FILE = "train.json"
WORD_DICTIONARY = "word_dict.txt"


def read_batch_data(start_line,
                    batch_size,
                    encoded_train_file=ENCODED_TRAINING_FILE):
    """Read one batch size data started from start_line.
    Args:
        start_line: the index for start line
        batch_size: the number for read data size
        encoded_train_file: require a json format file
    Returns:
        input_list: shape is [batch_size, time]
        label_list: shape is [batch_size]
    """
    input_list = []
    label_list = []
    with open(encoded_train_file, 'r') as train_obj:
        for line_index, json_str in enumerate(train_obj):
            if line_index == start_line-2+batch_size:
                break
            if line_index >= start_line-1:
                json_dict = json.load(json_str)
                input_list.append(json_dict['input'])
                label_list.append(json_dict['label'])

    return input_list, label_list


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
                      encoded_training_file=ENCODED_TRAINING_FILE):
    """Encoding training data
    Args:
        train_file: require to a json format file.
        encoded_training_file: a encoded training file.
    """

    encoded_training_obj = open(encoded_training_file, 'w', encoding="utf-8")

    # Read json file:
    with open(train_file, 'r', encoding="utf-8") as file_obj:
        for line in file_obj:
            decoded_dict = json.loads(line)
            encoded_dict = {}
            encoded_dict["label"] = decoded_dict["label"]



if __name__ == "__main__":
    """Set parameters."""
    batch_size = 10
    time = 12

    """Create data."""
    input = np.random.randint(
        low=1, high=547, size=(batch_size, time)
    )

    # Test
    # decode_and_write_data(input)

    encode_training_data()
