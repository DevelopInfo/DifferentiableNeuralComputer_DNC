# -*- coding: utf-8 -*-

import json

FILE_PATH = "corpus.txt"
DICT_NAME = "word_dict.txt"
STOP_WORD = "stop_word.txt"
TEST_PATH = "test.txt"
TEST_JSON_FILE = "test.json"
TRAIN_JSON_FILE = "train.json"


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


def txt_to_json(batch_size=-1,
                json_file=TRAIN_JSON_FILE,
                input_file=TEST_PATH,
                stop_word=STOP_WORD):
    train_obj = open(json_file, 'w', encoding='utf-8')

    # Create a list of stop word
    stop_word_list = []
    with open(stop_word, 'r') as stop_obj:
        for line in stop_obj:
            for word in line:
                stop_word_list.append(word)

    # # Get the max length for sentence.
    # max_sentence = 0
    # with open(input_file, 'r') as input_obj:
    #     for index, line in enumerate(input_obj):
    #         counter = 0
    #         for word in line:
    #             if word in stop_word_list:
    #                 continue
    #             counter += 1
    #             if counter > max_sentence:
    #                 max_sentence = counter
    #                 max_index = index
    #     print(max_index)

    # Encode data, padding data and create json file
    with open(input_file, 'r', encoding='utf-8') as input_obj:
        for index, line in enumerate(input_obj):
            if batch_size != -1 and index >= batch_size:
                break
            data_dict = {}
            data_dict["label"] = ""
            data_dict["input"] = ""
            counter = 0
            for word in line:
                if word in stop_word_list:
                    continue
                counter += 1
                data_dict["input"] = data_dict["input"] + word

            # # padding data
            # if counter < max_sentence:
            #     while counter < max_sentence:
            #         counter += 1
            #         data_dict["input"] = data_dict["input"] + "%d " % 0

            json_str = json.dumps(data_dict).encode('utf-8').decode('unicode-escape')
            train_obj.write(json_str + "\n")

    train_obj.close()


# ENCODED_TEST_FILE = "encoded_test.json"
# ENCODED_TRAINING_FILE = "encoded_train.json"

# def encodine_data(raw_file=TRAIN_JSON_FILE,
#                   encoded_file=ENCODED_TRAINING_FILE,
#                   dict_file=DICT_NAME,
#                   stop_file=STOP_WORD):
#     """Encoding training data
#     Args:
#         raw_file: require to a json format file.
#         encoded_file: a encoded training file is also a json format file.
#     """
#     # Create a dictionary.
#     dict = {}
#     dict_counter = 1
#     with open(dict_file, 'r', encoding="utf-8") as dict_obj:
#         for line in dict_obj:
#             for word in line:
#                 if word == "\n":
#                     continue
#                 dict[word] = dict_counter
#                 dict_counter += 1
#
#     # Create a list of stop word
#     stop_word_list = []
#     with open(stop_file, 'r', encoding="utf-8") as stop_obj:
#         for line in stop_obj:
#             for word in line:
#                 stop_word_list.append(word)
#
#     # Open the encoded training file
#     encoded_training_obj = open(encoded_file, 'w',
#                                 encoding="utf-8")
#     # Read json file:
#     with open(raw_file, 'r', encoding="utf-8") as file_obj:
#         for line in file_obj:
#             decoded_dict = json.loads(line)
#             encoded_dict = {}
#             encoded_dict["label"] = decoded_dict["label"]
#             encoded_dict["input"] = ""
#             for word in decoded_dict["input"]:
#                 if word != '0' and word in stop_word_list:
#                     continue
#                 if word == '0':
#                     encoded_dict["input"] = encoded_dict["input"] + "0 "
#                 else:
#                     encoded_dict["input"] = encoded_dict["input"] + "%d" % dict[word] + " "
#             encoded_str = json.dumps(encoded_dict)
#             encoded_training_obj.write(encoded_str+"\n")
#     encoded_training_obj.close()


if __name__ == "__main__":
    # create_dictoinary(FILE_PATH, DICT_NAME, STOP_WORD)

    # test.txt to train.json
    txt_to_json(batch_size=40, json_file=TRAIN_JSON_FILE)

    # test.txt to test.json
    txt_to_json(json_file=TEST_JSON_FILE)

    # encoding train.json to encoded_train.json
    # encodine_data(raw_file=TRAIN_JSON_FILE)

    # encoding text.json to encoded_test.json
    # encodine_data(raw_file=TEST_JSON_FILE)