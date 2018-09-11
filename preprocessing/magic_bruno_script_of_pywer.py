"""


The jargon used in the documentation of this code that might be non-trivial to infer:

The expected format of the prompt identifiers in the .mlf script file looks like:
"ABC111-XXXXX-XXXXXXXX-SA0001-en.lab"
of which:
ABC111 - is being referred to as location_id
SA0001 - is being referred to as section_id where A is the actual section

If a section is composed of an overall question with multiple subquestion, the section is being referred to as
multi-section. The overall section question that should prepend all the subquestion is referred to as master question.



Some other quirks:
If the prompt section id number is above that of a master section id given in the flag, it will be set to be that of
the flag.

-----

Generates files:
responses.txt prompts.txt speakers.txt conf.txt sections.txt prompt_ids.txt

"""
from __future__ import print_function
import sys
import os
import re
import time
import argparse

parser = argparse.ArgumentParser(description="We'll see what this actually does")  # todo

parser.add_argument('scripts_path', type=str, help='Path to a scripts (prompts) .mlf file')
parser.add_argument('responses_path', type=str, help='Path to a transcript of responses .mlf file')
parser.add_argument('save_dir', type=str, help='Path to the directory in which to save the processed files.')
parser.add_argument('--fixed_sections', nargs='*', type=str, default=['A', 'B'],
                    help='Which sections if any are fixed (questions are always the same). Takes space separated '
                         'indentifiers, e.g. --fixed_sections A B')
# todo: fixed_questions might not be needed
parser.add_argument('--fixed_questions', nargs='*', type=str, default=[],
                    help='Which individual questions if any are fixed (questions are always the same). Takes space separated '
                         'indentifiers, e.g. --fixed_questions SC0001')
parser.add_argument('--exclude_sections', nargs='*', type=str, default=['A', 'B'],
                    help='Sections to exclude from the output. '
                         'Takes space separated identifiers, e.g. --exclude_sections A B')
parser.add_argument('--multi_sections', nargs='*', type=str, default=['E'],
                    help='Which sections if any are multisections, i.e. when generating prompts a master question '
                         'should be pre-pended before each subquestion. Takes space separated '
                         'indentifiers, e.g. --multi_sections E F')
parser.add_argument('--multi_sections_master', nargs='*', type=str, default=['SE0006'],
                    help='The master question section id for each multi-section. For instance, if two sections '
                         'E and F are multi-sections with section ids SE0006 and SF0008 for the master or "parent"'
                         'questions to be prepended before subquestion, use:'
                         '--multi_sections_master SE0006 SF0008')
parser.add_argument('--speaker_grades_path', type=str, help='Path to a .lst file with speaker ids and their grades for each section', default='')
parser.add_argument('--num_sections', type=int, help='Number of sections in the test (e.g. in BULATS: A, B, C, D, E -> 5 sections).')
parser.add_argument('--exclude_grades_below', type=float, help='Exclude all the responses from sections with grades below a certain value', default=0.0)


sub_prompt_separator = "</s> <s>"


def generate_grade_dict(speaker_grade_path):
    """Crate a dict from speaker id to a dict that maps section to grade that the speaker got on this section."""
    grade_dict = dict()
    with open(speaker_grade_path, 'r') as file:
         for line in file.readlines():
             line = line.replace('\n', '').split('\t')
             speaker_id = line[0]
             section_dict = {}
             for i, section in zip(range(1, 7), ['A', 'B', 'C', 'D', 'E']):
                 # todo: replace with a non-manual labeling of sections
                 section_dict[section] = line[i]
             grade_dict[speaker_id] = section_dict
    return grade_dict


def extract_uniq_identifier(prompt_id, args):
    """
    Extract a unique identifier for each prompt.
    """
    location_id, section_id = prompt_id.split('-')

    # todo: See if this one is still needed now that dataset has been fixed
    ##### The manual rules: ###
    if (section_id[1] == 'C' or section_id[1] == 'D') and int(section_id[2:]) > 1:
        section_id = section_id[:2] + '0001'
        print("Prompt id: {} is being mapped to {} with manual rules.".format(prompt_id, '-'.join((location_id, section_id))))
    #####

    # If the prompt is from a fixed section or is a fixed question
    if section_id[1] in args.fixed_sections:
        return section_id

    # Consider whether the prompt is a master question (must be exclusive from fixed section
    if section_id[1] in args.multi_sections:
        idx = args.multi_sections.index(section_id[1])
        # Assume if question number is larger than that of a master question, it is a master question
        if int(section_id[2:]) > int(args.multi_sections_master[idx][2:]):
            section_id = args.multi_sections_master[idx]

    return '-'.join((location_id, section_id))


def generate_mappings(prompts, prompt_ids, args):
    """
    Assumes the mapping from prompt_ids to prompts is surjective, but not necessarily injective: many prompt_ids
    can point to the same prompt. Hence in the inverse mapping, each prompt points to a list of prompt_ids that point
    to it.

    The prompt id's take the form:
    prompt_id = 'ABC999-SA0001' where ABC999 is the location_id and SA0001 the question number:
    A indicates the section, and 0001 indicates question 1 within that section.

    For prompts that have been marked with a fixed_sections flag, i.e. the corresponding question of each section is
    always the same, we store additional prompt_id key pointing to the prompt which is just the section identifier
    without the location identifier, i.e.:
    If 'A' is in fixed_section:
    'ABC999-SA0001' -> prompt1
    and also:
    'SA0001' -> prompt1

    :return: mapping dict, inv_mapping dict
    """

    mapping = {}  # Mapping from prompts to a list of identifiers
    inv_mapping = {}  # Mapping from identifiers to prompts

    for prompt, full_id in zip(prompts, prompt_ids):
        # Process the prompt_id line
        assert re.match(r'[A-Z0-9]+-[A-Z0-9]+-[a-z]*$', full_id)  # The expected format of the line

        full_id = full_id.split('-')
        location_id, section_id = full_id[0], full_id[-2]
        prompt_id = '-'.join((location_id, section_id))
        _add_pair_to_mapping(mapping, inv_mapping, prompt_id, prompt)
        _add_pair_to_mapping(mapping, inv_mapping, extract_uniq_identifier(prompt_id, args), prompt)

    return mapping, inv_mapping


def _add_pair_to_mapping(mapping, inv_mapping, prompt_id, prompt):
    mapping.setdefault(prompt, [])
    mapping[prompt].append(prompt_id)
    inv_mapping[prompt_id] = prompt
    return


def process_mlf_scripts(mlf_path, word_pattern=r"[%A-Za-z'\\_.]+$"):
    sentences = []
    ids = []
    with open(mlf_path, 'r') as file:
        words = []
        for line in file.readlines():
            line = line.strip()  # Remove the \n at the end of line
            if line == '#!MLF!#':
                # Ignore the file type prefix line
                continue
            elif ".lab" in line:
                assert re.match(r'"[a-zA-Z0-9-]+.lab"$', line)
                assert len(sentences) == len(ids)
                # Get rid of the " at the beginning and the .lab" at the end
                line = line[1:-5]
                ids.append(line)
            elif line == ".":
                # A "." indicates end of utternace -> add the sentence to list
                sentence = " ".join(words)
                assert len(sentence) > 0
                sentences.append(sentence)
                # Reset the words temporary list
                words = []
            elif re.match(word_pattern, line):
                words.append(line)
            else:
                raise ValueError("Unexpected pattern in file: " + line)
    return sentences, ids


def process_mlf_responses(mlf_path, word_line_pattern=r"[0-9]* [0-9]* [\"%A-Za-z'\\_.]+ [0-9.]*$"):
    sentences = []
    ids = []
    confidences = []

    with open(mlf_path, 'r') as file:
        words_temp = []
        confs_temp = []
        for line in file.readlines():
            line = line.strip()  # Remove the \n at the end of line
            if line == '#!MLF!#':
                # Ignore the file type prefix line
                continue
            elif ".lab" in line:
                assert re.match(r'"[a-zA-Z0-9-_]+.lab"$', line)
                assert len(sentences) == len(ids)
                # Get rid of the " at the beginning and the .lab" at the end
                line = line[1:-5]
                ids.append(line)
            elif line == ".":
                # A "." indicates end of utternace -> add the sentence to list
                sentence = " ".join(words_temp)
                sentence_confs = " ".join(confs_temp)
                assert len(sentence) > 0
                sentences.append(sentence)
                confidences.append(sentence_confs)
                # Reset the temp variables
                words_temp, confs_temp = [], []
            elif len(line.split()) > 1:
                # assert re.match(word_line_pattern, line)
                line_split = line.split()
                words_temp.append(line_split[-2])
                confs_temp.append(line_split[-1])
            else:
                raise ValueError("Unexpected pattern in file: " + line)
    return sentences, ids, confidences


def process_responses_file(responses, full_ids, confidences, inv_mapping, args, grade_dict=None):
    speakers = []
    sections = []
    prompt_ids = []
    prompts = []
    grades = []

    i = 0
    while i < len(responses):
        full_id = full_ids[i].split('-')
        location_id = full_id[0]
        speaker_number = full_id[1]
        section_id = full_id[3]
        section = section_id[1]

        if section in args.exclude_sections:
            del responses[i], full_ids[i], confidences[i]
            continue
        else:
            speaker = '-'.join((location_id, speaker_number))  # Should extract the speaker id
            prompt_id = '-'.join((location_id, section_id))
            prompt = inv_mapping.get(extract_uniq_identifier(prompt_id, args), None)
            if prompt is None:
                print("Didn't find a match for prompt id: ", prompt_id)
                del responses[i], full_ids[i], confidences[i]
                continue
            # If processing the grades as well:
            if grade_dict is not None:
                try:
                    grade = grade_dict[speaker][section]
                except KeyError:
                    print("No grade for speaker {}".format(speaker))
                    # Set the prompt grade to empty
                    grade = ''
                if args.exclude_grades_below > 0.:
                    if (grade == '' or grade =='.'):
                        del responses[i], full_ids[i], confidences[i]
                        continue
                    elif float(grade) < args.exclude_grades_below:
                        del responses[i], full_ids[i], confidences[i]
                        continue
                else:
                    grades.append(grade)

            # Store the relevant data
            speakers.append(speaker)
            prompt_ids.append(prompt_id)
            prompts.append(prompt)
            sections.append(section)
        i += 1

    # Assert number of elements of  responses, response confidences, speakers, and prompt_ids is the same
    assert len({len(responses), len(confidences), len(speakers), len(prompt_ids), len(prompts), len(sections)}) == 1

    return responses, full_ids, confidences, prompt_ids, prompts, speakers, sections, grades


def fixed_section_filter(prompt_id, fixed_sections, fixed_questions):
    section_id = prompt_id.split('-')[1]  # Extracts the section id, for example 'SA0001'
    if section_id[1] in fixed_sections or section_id in fixed_questions:
        print("Id lookup reduced: , ", prompt_id)
        return section_id
    else:
        return prompt_id


def main(args):
    # Check the input flag arguments are correct:
    try:
        for section in args.fixed_sections + args.exclude_sections + args.multi_sections:
            assert re.match(r'[A-Z]', section)
        for section_id in args.multi_sections_master:
            assert re.match(r'[A-Z0-9]', section_id)
        assert len(args.multi_sections) == len(args.multi_sections_master)
    except AssertionError:
        raise ValueError(
            "The flag arguments provided don't match the expected format. See the manual for expected arguments.")

    # Cache the command:
    if not os.path.isdir(os.path.join(args.save_dir, 'CMDs')):
        os.makedirs(os.path.join(args.save_dir, 'CMDs'))
    with open(os.path.join(args.save_dir, 'CMDs/preprocessing.cmd'), 'a') as f:  
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    start_time = time.time()
    # Process the prompts (scripts) file
    prompts_list, prompt_ids_list = process_mlf_scripts(args.scripts_path)
    print('Prompts script file processed. Time eta: ', time.time() - start_time)

    # Generate mappings
    mapping, inv_mapping = generate_mappings(prompts_list, prompt_ids_list, args)

    print("Mappings generated. Time elapsed: ", time.time() - start_time)
    
    # Generate the grades dictionary
    if args.speaker_grades_path:
        grades_dict = generate_grade_dict(args.speaker_grades_path)
        print('Generated the speaker grades dictionary.')

    # Process the responses
    responses, full_ids, confidences = process_mlf_responses(args.responses_path)
    print("Responses mlf file processed. Time elapsed: ", time.time() - start_time)
    if args.speaker_grades_path:
        responses, full_ids, confidences, prompt_ids, prompts, speakers, sections, grades = process_responses_file(responses,
                                                                                                       full_ids,
                                                                                                       confidences,
                                                                                                       inv_mapping,
                                                                                                       args, grades_dict)
    else:
        responses, full_ids, confidences, prompt_ids, prompts, speakers, sections, _ = process_responses_file(responses,
                                                                                                       full_ids,
                                                                                                       confidences,
                                                                                                       inv_mapping,
                                                                                                       args)

    print("Responses transcription processed. Time elapsed: ", time.time() - start_time)


    # Handle the multi subquestion prompts
    for i in range(len(sections)):
        if sections[i] in args.multi_sections:
            # Get the section_id of the master question
            master_section_id = args.multi_sections_master[args.multi_sections.index(sections[i])]
            # Get the whole prompt id of the master question
            location_id = prompt_ids[i].split('-')[0]
            master_prompt_id = '-'.join((location_id, master_section_id))

            # Prepend the master prompt to the subquestion prompts.
            try:
                subquestion_prompt = prompts[i]
                master_prompt = inv_mapping[master_prompt_id]
                new_prompt = ' '.join([master_prompt, sub_prompt_separator, subquestion_prompt])
                prompts[i] = new_prompt
            except KeyError:
                print("No master question: " + master_prompt_id + " in the scripts file")
                # todo: Potentially delete something

    # Write the data to the save directory:
    suffix = '.txt'

    # Make sure the directory exists:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for data, filename in zip([responses, confidences, speakers, prompts, prompt_ids, sections],
                              ['responses', 'confidences', 'speakers', 'prompts', 'prompt_ids', 'sections']):
        file_path = os.path.join(args.save_dir, filename + suffix)
        with open(file_path, 'w') as file:
            file.write('\n'.join(data))
    if args.speaker_grades_path:
        file_path = os.path.join(args.save_dir, 'grades' + suffix)
        with open(file_path, 'w') as file:
            file.write('\n'.join(grades))

    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
