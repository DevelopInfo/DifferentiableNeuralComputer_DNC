# -*- coding: utf-8 -*-

FILE_PATH = "corpus.txt"
DICT_NAME = "word_dict.txt"
STOP_WORD = "stop_word.txt"
TEST_PATH = "test.txt"
ENCODING_FILE = "encoding.txt"


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


def encoding_data(input_file,
                  encoding_file,
                  dict_file,
                  stop_word):
    """Make an encoding file"""
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

    with open(input_file, 'r') as input_obj:
        for line in input_obj:
            data_line = ""
            for word in line:
                if word in stop_word_list:
                    continue
                # print(dict[word])
                data_line = data_line + "%d" % dict[word] + " "
            encoding_file.write(data_line+"\n")

    encoding_file.close()




if __name__ == "__main__":
    # create_dictoinary(FILE_PATH, DICT_NAME, STOP_WORD)
    # train_to_test(FILE_PATH, TEST_PATH)
    encoding_data(TEST_PATH, ENCODING_FILE, DICT_NAME, STOP_WORD)
    pass