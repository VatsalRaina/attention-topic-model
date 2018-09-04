"""

-----

Generates files:
responses.txt prompts.txt speakers.txt conf.txt sections.txt

"""
from __future__ import print_function
import sys
import os
import re
import argparse

parser = argparse.ArgumentParser(description="We'll see what this actually does")  # todo

parser.add_argument('script_path', type=str, help='Path to a script (prompts) .mlf file')
parser.add_argument('response_path', type=str, help='Path to a transcript of responses .mlf file')
parser.add_argument('--exclude_AB', type=bool, default=True,
                    help='Whether to exclude section A and B')



def main(args):
    # Process the prompts

    words = []
    confidences = []
    with open(args.script_path, 'r') as script:
        for line in script.readlines():
            # Ignore the file type prefix line
            if line.strip() == '#!MLF!#':
                continue

            line = line.replace('\n', '').split()
            if len(line) > 1:
                wrd = line[-2]
                conf = line[-1]
            else:
                if line[0] == '.':
                    wrd = '</s>\n'
                    conf = '1.0\n'
                else:
                    wrd = '<s>'
                    conf = '1.0'
            words.append(wrd)
            confidences.append(conf)

    text = ' '.join(words)
    confidences = ' '.join(confidences)
    with open(args.TXT + '.dat', 'w') as handle:
        handle.write(text)
    with open(args.TXT + '.conf', 'w') as handle:
        handle.write(confidences)


    # Create mapping dict from script mlf file

    tmp = {}
    q_dict = {}
    a_dict = {}
    with open('script-maps/scripts-' + args.section + '-map.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split()

            if line[1][0:2] == 'SA':
                if line[1] in ['SA0016',  # ...
                               'SA0121']:
                    value = ' '.join(line[0:2])
                else:
                    value = line[1]
            else:
                if line[1][0:2] == 'SC':
                    line[1] = "SC0001"
                elif line[1][0:2] == 'SD':
                    line[1] = "SD0001"
                value = ' '.join(line[0:2])
            key = ' '.join(line[2:])
            if tmp.has_key(key):
                tmp[key].append(value)
            else:
                tmp[key] = [value]

        for item, i in zip(tmp.items(), xrange(len(tmp.items()))):
            #       print item[0], item[1], i
            q_dict[i] = item[0]
            for key in item[1]:
                a_dict[key] = i

                # print q_dict
        path = 'qa_' + args.section + '_dicts.pickle'
        with open(path, 'wb') as handle:
            pickle.dump([q_dict, a_dict], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Create inverse mapping

    #
    # Write to files:
    for ... :
        # Write to the respones.txt
        # Write to..
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
