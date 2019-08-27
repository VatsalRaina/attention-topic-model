#! /usr/bin/env python

"""
Converts processed data into equivalent processed data with specified prompt-response pairs removed from the data
"""

import os
import shutil


def main():

    load_path = '/home/miproj/urop.2019/vr311/data_vatsal/BULATS/prompts.txt'
    dict_prompt_nums = {}

    f = open(load_path, "r")
    all_samples = f.readlines()
    all_samples_imp = [line.rstrip('\n') for line in all_samples]
    f.close()

    line_num = 1
    for sample in all_samples_imp:
        if sample in dict_prompt_nums:
            dict_prompt_nums[sample][0]+=1
            dict_prompt_nums[sample][1].append(line_num)
        else:
            dict_prompt_nums[sample]=[1,[line_num]]
        line_num+=1

    # Find lines of prompt of interest
    for value in dict_prompt_nums.values():
        #print(value)
        if value[0] == 12395:
            lines_to_remove = value[1]
            break

    lines_to_remove.reverse()
   # print(lines_to_remove)
   
    source_dir = '/home/miproj/urop.2019/vr311/data_vatsal/BULATS' 
    destination_dir = '/home/miproj/urop.2019/vr311/data_vatsal/BULATS/ID_0__removed_12395'

    shutil.copyfile(os.path.join(source_dir, 'prompts.txt'), os.path.join(destination_dir, 'prompts.txt'))
    shutil.copyfile(os.path.join(source_dir, 'prompt_ids.txt'), os.path.join(destination_dir, 'prompt_ids.txt'))
    shutil.copyfile(os.path.join(source_dir, 'responses.txt'), os.path.join(destination_dir, 'responses.txt'))
    shutil.copyfile(os.path.join(source_dir, 'speakers.txt'), os.path.join(destination_dir, 'speakers.txt'))
    shutil.copyfile(os.path.join(source_dir, 'confidences.txt'), os.path.join(destination_dir, 'confidences.txt'))
    shutil.copyfile(os.path.join(source_dir, 'sections.txt'), os.path.join(destination_dir, 'sections.txt'))
    shutil.copyfile(os.path.join(source_dir, 'grades.txt'), os.path.join(destination_dir, 'grades.txt'))

    lines_prompts = open(os.path.join(source_dir, 'prompts.txt'), 'r').readlines()
    lines_prompt_ids = open(os.path.join(source_dir, 'prompt_ids.txt'), 'r').readlines()
    lines_responses = open(os.path.join(source_dir, 'responses.txt'), 'r').readlines()
    lines_speakers = open(os.path.join(source_dir, 'speakers.txt'), 'r').readlines()
    lines_confidences = open(os.path.join(source_dir, 'confidences.txt'), 'r').readlines()
    lines_sections = open(os.path.join(source_dir, 'sections.txt'), 'r').readlines()
    lines_grades = open(os.path.join(source_dir, 'grades.txt'), 'r').readlines()    

    for curr in lines_to_remove:
        lines_prompts[curr-1] = ''
        lines_prompt_ids[curr-1] = ''
        lines_responses[curr-1] = ''
        lines_speakers[curr-1] = ''
        lines_confidences[curr-1] = ''
        lines_sections[curr-1] = ''
        lines_grades[curr-1] = ''

    out_prompts = open(os.path.join(destination_dir, 'prompts.txt'), 'w')
    out_prompts.writelines(lines_prompts)
    out_prompts.close()

    out_prompt_ids = open(os.path.join(destination_dir, 'prompt_ids.txt'), 'w')
    out_prompt_ids.writelines(lines_prompt_ids)
    out_prompt_ids.close()
    
    out_responses = open(os.path.join(destination_dir, 'responses.txt'), 'w')
    out_responses.writelines(lines_responses)
    out_responses.close()

    out_speakers = open(os.path.join(destination_dir, 'speakers.txt'), 'w')
    out_speakers.writelines(lines_speakers)
    out_speakers.close()
    
    out_confidences = open(os.path.join(destination_dir, 'confidences.txt'), 'w')
    out_confidences.writelines(lines_confidences)
    out_confidences.close()

    out_sections = open(os.path.join(destination_dir, 'sections.txt'), 'w')
    out_sections.writelines(lines_sections)
    out_sections.close()
    
    out_grades = open(os.path.join(destination_dir, 'grades.txt'), 'w')
    out_grades.writelines(lines_grades)
    out_grades.close()


if __name__ == '__main__':
    main()
