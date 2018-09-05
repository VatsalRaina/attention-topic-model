"""


The jargon used in the documentation of this code that might be non-trivial to infer:

The expected format of the prompt identifiers in the .mlf script file looks like:
"ABC111-XXXXX-XXXXXXXX-SA0001-en.lab"
of which:
ABC111 - is being referred to as location_id
SA0001 - is being referred to as section_id where A is the actual section

For the responses the expected format is:
# todo

If a section is composed of an overall question with multiple subquestion, the section is being referred to as
multi-section. The overall section question is referred to as master question.



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


sub_prompt_separator = " </s> <s> "


def process_prompts_file(scripts_path, fixed_sections, multi_sections_master):
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

    with open(scripts_path, 'r') as file:
        prompt_words = []
        for line in file.readlines():
            line = line.strip()  # Remove the \n at the end of line
            if line == '#!MLF!#':
                # Ignore the file type prefix line
                continue
            elif ".lab" in line:
                assert re.match(r'"[A-Z0-9]*-[A-Z0-9]*-[A-Z0-9]*-[A-Z0-9]*-[a-z]*.lab"$', line)
                line = line[1:-5]  # Ignore the " at the beginning and .lab" at the end
                line = line.split('-')
                location_id = line[0]
                section_id = line[-2]

                prompt_id = '-'.join((location_id, section_id))
            elif line == ".":
                # A "." indicates end of prompt -> add the prompt to the mapping dictionaries
                prompt = " ".join(prompt_words)
                assert prompt_id is not None
                assert len(prompt) > 0
                mapping.setdefault(prompt, [])
                mapping[prompt].append(prompt_id)
                inv_mapping[prompt_id] = prompt

                # Handle the fixed_sections
                location_id, section_id = prompt_id.split('-')  # Extracts the section id, for example 'SA0001'
                if section_id[1] in fixed_sections:
                    inv_mapping[section_id] = prompt

                # Consider whether the prompt is a master question
                if section_id[1] in multi_sections:
                    idx = multi_sections.index(section_id[1])
                    # Assume if question number is larger than that of a master question, it is a master question
                    if int(section_id[2:]) > int(multi_sections_master[idx][2:]):
                        inv_mapping['-'.join(location_id, multi_sections_master[idx])] = prompt

                # Reset the variables for storing the elements of the prompt
                prompt_id = None
                prompt_words = []
            elif re.match(r"[%A-Za-z'\\_.]+$", line):
                prompt_words.append(line)
            else:
                raise ValueError("Unexpected pattern in file: " + line)

    return mapping, inv_mapping


def prepend_multiquestion_master(prompts, sections, multi_section='SE', master_question='SE006'):
    for i in range(len(sections)):
        if sections[i] == multi_section:
            pass
    return


def process_responses_file(responses_path, exclude_sections):
    """
    :param responses_path: path to the file with the transcriptions of responses
    :param exclude_sections: list of sections to exclude, e.g. ['A', 'B']
    :return: responses, confidences, speakers, prompt_ids
    """
    responses = []
    confidences = []
    speakers = []
    prompt_ids = []

    response_words = []  # Temporarily stores words for each response before adding to list
    confidences_temp = []  # Temporarily stores confidences for each response before adding to list

    skip_response = False
    with open(responses_path, 'r') as file:
        for line in file.readlines():
            line = line.strip()

            if skip_response:
                if line == ".":
                    skip_response = False
                continue

            # Ignore the file type prefix line
            if line == '#!MLF!#':
                continue
            elif ".lab" in line:
                # Speaker and prompt identifier
                assert re.match(r'"[a-zA-Z0-9_-]*.lab"$', line)

                # Extract relevant information from identifier
                line = line[1:-5]  # Get rid of " at the beginning and .lab" at the end
                line = line.split('-')
                location_id = line[0]
                speaker_number = line[1]
                section_id = line[3]

                # Check if this response should be skipped
                if section_id[1] in exclude_sections:
                    skip_response = True
                    continue

                speaker = '-'.join((location_id, speaker_number))  # Should extract the speaker id
                prompt_id = '-'.join((location_id, section_id))

                # Store the relevant 'metadata'
                speakers.append(speaker)
                prompt_ids.append(prompt_id)
            elif line == ".":
                # A "." indicates end of response -> add the response to the list
                response = " ".join(response_words)
                response_conf = " ".join(confidences_temp)
                assert len(confidences_temp) == len(response_words)
                assert len(response) > 0

                responses.append(response)
                confidences.append(response_conf)

                # Reset the variables for storing the elements of the prompt
                response_words = []
                confidences_temp = []
            elif len(line.split()) > 1:
                if re.match(r"[0-9]* [0-9]* [\"%A-Za-z'\\_.]+ [0-9.]*$", line):
                    pass
                elif re.match(r"[\"%A-Za-z'\\_.]+ [0-9.]*$", line):
                        # this should match the eval data
                    pass
                else:
                    raise ValueError("Unexpected line: ", line)
                line = line.split()
                word = line[-2]
                conf = line[-1]
                response_words.append(word)
                confidences_temp.append(conf)
            else:
                raise ValueError("Unexpected pattern in file: " + line)

    # Assert number of elements of  responses, response confidences, speakers, and prompt_ids is the same
    assert len({len(responses), len(confidences), len(speakers), len(prompt_ids)}) == 1

    return responses, confidences, speakers, prompt_ids


def fixed_section_filter(prompt_id, fixed_sections):
    section_id = prompt_id.split('-')[1]  # Extracts the section id, for example 'SA0001'
    if section_id[1] in fixed_sections:
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

    start_time = time.time()
    # Process the prompts
    mapping, inv_mapping = process_prompts_file(args.scripts_path, args.fixed_sections, args.multi_sections_master)

    print("Prompt script file processed. Mappings generated. Time elapsed: ", time.time() - start_time)

    # todo: print mapping to see if correct
    # todo: possibly store the mapping

    # Process the responses
    responses, confidences, speakers, prompt_ids = process_responses_file(args.responses_path, args.exclude_sections)
    print("Responses transcription processed. Time elapsed: ", time.time() - start_time)

    # Extract the section data for each response (A, B, C, ...)
    sections = list(map(lambda prompt_id: prompt_id.split('-')[1][1], prompt_ids))

    # Reduce prompt ids for fixed section to section ids:
    prompt_ids_red = map(lambda prompt_id: fixed_section_filter(prompt_id, args.fixed_sections), prompt_ids)

    # Generate the prompts list (in the same order as responses)
    prompts = list(map(lambda prompt_id: inv_mapping.setdefault(prompt_id, None), prompt_ids_red))

    print('Any unmapped: ', None in prompts)
    # todo: Filter the responses, prompts, ... e.t.c. where prompts is None

    # Handle the multi subquestion prompts
    for i in range(len(sections)):
        if sections[i] in args.multi_sections:
            # Get the section_id of the master question
            master_section_id = args.multi_sections_master[args.multi_sections.index(sections[i])]
            # Get the whole prompt id of the master question
            location_id = prompt_ids[i].split('-')[0]
            master_prompt_id = '-'.join((location_id, master_section_id))

            # Prepend the master prompt to the subquestion prompts.
            prev_prompt = prompts[i]
            master_prompt = inv_mapping[master_prompt_id]
            new_prompt = master_prompt + sub_prompt_separator + prev_prompt
            prompts[i] = new_prompt

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
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
