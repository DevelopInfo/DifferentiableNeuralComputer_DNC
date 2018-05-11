# -*- coding: utf-8 -*-

import sys
import os


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
        story = []
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
                line_encode = []
                for word_index, word in enumerate(line.split()):
                    if word.isalpha():
                        line_encode.append(lexicons_dictionary[word.lower()])
                    # the tag to begin story
                    if word_index == 0 and word == '1':
                        story = []
                    if word == '?':
                        query_tag = True
                        query_file.writelines(line_encode)
                    if query_tag and word.isalpha():
                        answer_file.writelines(lexicons_dictionary[word.lower()])
                    if query_tag and word.isalnum():
                        ans_index_file.writelines(word)
                if not query_tag:
                    story.append(line_encode)
                else:
                    story_file.writelines(story)
                    query_tag=False

        story_file.close()
        query_file.close()
        answer_file.close()
        ans_index_file.close()




lexicons_dict = create_dictionary("./tasks_1-20_v1-2/en/")
encode_data("./tasks_1-20_v1-2/en/", lexicons_dict)