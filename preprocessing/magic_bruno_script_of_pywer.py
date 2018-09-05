"""

-----

Generates files:
responses.txt prompts.txt speakers.txt conf.txt sections.txt prompt_ids.txt

"""
from __future__ import print_function
import sys
import os
import re
import argparse

parser = argparse.ArgumentParser(description="We'll see what this actually does")  # todo

parser.add_argument('scripts_path', type=str, help='Path to a scripts (prompts) .mlf file')
parser.add_argument('responses_path', type=str, help='Path to a transcript of responses .mlf file')
parser.add_argument('save_dir', type=str, help='Path to the directory in which to save the processed files.')
parser.add_argument('--exclude_AB', type=bool, default=True,
                    help='Whether to exclude section A and B')

word_match = r"[%A-Za-z'\\._]"

def process_prompts_file(scripts_path):
    """
    Assumes the mapping from prompt_ids to prompts is surjective, but not necessarily injective: many prompt_ids
    can point to the same prompt. Hence in the inverse mapping, each prompt points to a list of prompt_ids that point
    to it.
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
                line = line[1:-5] # Ignore the " at the beginning and .lab" at the end
                line = line.split('-')
                location_id = line[0]
                section = line[-2]
                prompt_id = '-'.join((location_id, section))
            elif line == ".":
                # A "." indicates end of prompt -> add the prompt to the mapping dictionaries
                prompt = " ".join(prompt_words)
                assert prompt_id is not None
                assert len(prompt) > 0
                mapping.setdefault(prompt, [])
                mapping[prompt].append(prompt_id)
                inv_mapping[prompt_id] = prompt

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


def process_responses_file(responses_path):
    responses = []
    confidences = []
    speakers = []
    prompt_ids = []

    response_words = []  # Temporarily stores words for each response before adding to list
    confidences_temp = []  # Temporarily stores confidences for each response before adding to list
    with open(responses_path, 'r') as file:
        for line in file.readlines():
            line = line.strip()
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
                section = line[3]
                speaker = '-'.join((location_id, speaker_number))  # Should extract the speaker id
                prompt_id = '-'.join((location_id, section))

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
                try:  # todo: remove once confident in the format
                    assert re.match(r"[0-9]* [0-9]* [\"%A-Za-z'\\_.]+ [0-9.]*$", line)
                except AssertionError as e:
                    print(line)
                    raise e
                line = line.split()
                word = line[-2]
                conf = line[-1]
                response_words.append(word)
                confidences_temp.append(conf)
            else:
                raise ValueError("Unexpected pattern in file: " + line)

    # Assert number of elements of  responses, response confidences, speakers, and prompt_ids is the same
    # todo: remove the print functions once confident in script
    print(len(responses))
    print(len(confidences))
    print(len(speakers))
    print(len(prompt_ids))
    assert len({len(responses), len(confidences), len(speakers), len(prompt_ids)}) == 1

    return responses, confidences, speakers, prompt_ids


def main(args):
    # Process the prompts
    mapping, inv_mapping = process_prompts_file(args.scripts_path)
    print("Prompt script file processed. Mappings generated.")

    # todo: print mapping to see if correct
    # todo: possibly store the mapping

    # Process the responses
    responses, confidences, speakers, prompt_ids = process_responses_file(args.responses_path)
    print("Responses transcription processed.")

    # Generate the prompts list (in the same order as responses)
    prompts = map(lambda prompt_id: inv_mapping[prompt_id], prompt_ids)


    # Extract the section data for each response (SA, SB, SC, ...)
    sections = map(lambda prompt_id: prompt_id.split('-')[1][:2], prompt_ids)

    # Handle the multi subquestion prompts

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
