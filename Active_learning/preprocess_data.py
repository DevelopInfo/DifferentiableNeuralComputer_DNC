# -*- coding: utf-8 -*-

import json

FILE_PATH = "corpus.txt"
DICT_NAME = "word_dict.txt"
STOP_WORD = "stop_word.txt"
TEST_PATH = "test.txt"
ENCODING_FILE = "encoded_test.json"


def create_dictoinary(data_file, dict_file, stop_word):
    """Create a word dictionary file"""
    stop_list = []
    with open(stop_word, 'r') as file_obj:
        for line in file_obj:
            for word in line:
                stop_list.append(word)

    word_dict_file = open(dict_file, 'w')
    word_dict = []
    word_counter = 1
    with open(data_file, 'r') as file_obj:
        for line in file_obj:
            for word in line:
                if word in stop_list:
                    continue
                if word not in word_dict:
                    word_dict_file.write(word+"\n")
                    word_dict[word] = word_counter
                    word_counter += 1

    word_dict_file.close()


def train_to_test(train_file, test_file):
    """Make a test file from a training file"""
    test_obj = open(test_file, 'w')

    with open(train_file, 'r') as train_obj:
        for line in train_obj:
            for word in line:
                if word == "\t":
                    test_obj.write("\n")
                    break
                test_obj.write(word)

    test_obj.close()


def encoding_data(input_file=TEST_PATH,
                  encoding_file=ENCODING_FILE,
                  dict_file=DICT_NAME,
                  stop_word=STOP_WORD):
    """Make an encoding file
    Args:
        input_file: input_file is a common file.
        encoding_file: encoding_file is a json format file and is encoded.
    """
    encoding_file = open(encoding_file, 'w')

    # Create a dictionary.
    dict = {}
    dict_counter = 1
    with open(dict_file, 'r') as dict_obj:
        for line in dict_obj:
            for word in line:
                if word == "\n":
                    continue
                dict[word] = dict_counter
                dict_counter += 1

    # Create a list of stop word
    stop_word_list = []
    with open(stop_word, 'r') as stop_obj:
        for line in stop_obj:
            for word in line:
                stop_word_list.append(word)

    # Get the max length for sentence.
    max_sentence = 0
    with open(input_file, 'r') as input_obj:
        for line in input_obj:
            counter = 0
            for word in line:
                if word in stop_word_list:
                    continue
                counter += 1
                if counter > max_sentence:
                    max_sentence = counter

    # Encode data, padding data and create json file
    with open(input_file, 'r') as input_obj:
        for line in input_obj:
            data_dict = {}
            data_dict["label"] = ""
            data_dict["input"] = ""
            counter = 0
            for word in line:
                if word in stop_word_list:
                    continue
                counter += 1
                data_dict["input"] = data_dict["input"] + "%d " % int(dict[word])

            # padding data
            if counter < max_sentence:
                while counter < max_sentence:
                    counter += 1
                    data_dict["input"] = data_dict["input"] + "%d " % 0
            json_str = json.dumps(data_dict)
            encoding_file.write(json_str+"\n")

    encoding_file.close()


if __name__ == "__main__":
    # create_dictoinary(FILE_PATH, DICT_NAME, STOP_WORD)
    # train_to_test(FILE_PATH, TEST_PATH)
    encoding_data(TEST_PATH, ENCODING_FILE, DICT_NAME, STOP_WORD)
    pass