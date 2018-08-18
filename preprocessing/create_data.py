# !/usr/bin/python

import sys
import os
import re
import argparse

try:
    import cPickle as pickle
except:
    import pickle

SA_LIST = ['SA0016',
           'SA0030',
           'SA0032',
           'SA0034',
           'SA0041',
           'SA0042',
           'SA0056',
           'SA0062',
           'SA0066',
           'SA0071',
           'SA0079',
           'SA0081',
           'SA0121']

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('--MERGE_SE', type=bool, default=False,
                               help='Use Merged SE scripts')
commandLineParser.add_argument('dataset', type=str,
                               help='Name of the dataset to process')
commandLineParser.add_argument('section', type=str, choices=['SA', 'SB', 'SC', 'SD', 'SE'], default='SC',
                               help='Which section to process')


def main(argv=None):
    args = commandLineParser.parse_args()
    if args.MERGE_SE:
        path = '/home/alta/BLTSpeaking/exp-am969/attention_grader/qa_ALL_dicts.pickle'
        q_path = '_questions_merged.txt'
    else:
        q_path = '_questions.txt'
        path = '/home/alta/BLTSpeaking/exp-am969/attention_grader/qa_' + args.section + '_dicts.pickle'
    with open(path, 'rb') as handle:
        q_dict, a_dict = pickle.load(handle)

    # path = 'features_' + args.dataset + '_dict.pickle'
    #  with open(path, 'rb') as handle:
    #     feature_dict = pickle.load(handle)
    #lines = []

    missing = []
    try:
        with open(args.dataset + '_' + args.section + '.conf') as conf_raw:
            with open(args.section + '_conf.txt', 'w') as conf:
                with open(args.dataset + '_' + args.section + '.dat') as g:
                    with open(args.section + '_data.txt', 'w') as t:
                        with open('targets_' + args.section + '.tmp.txt', 'r') as f:
                            with open('targets_' + args.section + '.txt', 'w') as h:
                                with open('speakers_' + args.section + '.txt', 'w') as s:
                                    # with open('features_'+args.section+'.txt', 'w') as m:
                                    with open('topics_' + args.section + '.txt', 'w') as n:
                                        with open(args.section + q_path, 'w') as d:
                                            with open('targets_' + args.section + '_bulats.tmp.txt', 'r') as bt:
                                                with open('targets_' + args.section + '_bulats.txt', 'w') as b:
                                                    for line, grd, grd_bulats, conf_score in zip(g.readlines(),
                                                                                                 f.readlines(),
                                                                                                 bt.readlines(),
                                                                                                 conf_raw.readlines()):
                                                        grd = grd.replace('\n', '')
                                                        grd_bulats = grd_bulats.replace('\n', '')
                                                        line = line.replace('\n', '')
                                                        conf_line = conf_score.replace('\n', '')
                                                        line, conf_line = zip(*((wrd, conf) for wrd, conf in
                                                                                zip(line.split(), conf_line.split()) if
                                                                                not re.match(
                                                                                    '(%HESITATION%)|(\S*_%partial%)',
                                                                                    wrd)))
                                                        if line[2][0:2] == 'SA':
                                                            if line[2] in SA_LIST:
                                                                topic = ' '.join(line[1:3])
                                                            else:
                                                                topic = line[2]
                                                        else:
                                                            topic = ' '.join(line[1:3])
                                                        try:
                                                            if len(line[3:]) > 2 and len(conf_line[3:]) > 2:
                                                                text = ' '.join(line[3:])
                                                                conf_scores = ' '.join(conf_line[3:])
                                                                question = q_dict[a_dict[topic]]
                                                                # feats = feature_dict[line[0]]
                                                                d.write(question + '\n')
                                                                h.write(grd + '\n')
                                                                t.write(text + '\n')
                                                                conf.write(conf_scores + '\n')
                                                                s.write(line[0] + '\n')
                                                                # m.write(feats+'\n')
                                                                n.write(topic + '\n')
                                                                b.write(grd_bulats + '\n')
                                                        except:
                                                            missing.append(topic)
    except:
        with open(args.dataset + '_' + args.section + '.conf') as conf_raw:
            with open(args.section + '_conf.txt', 'w') as conf:
                with open(args.dataset + '_' + args.section + '.dat') as g:
                    with open(args.section + '_data.txt', 'w') as t:
                        with open('targets_' + args.section + '.tmp.txt', 'r') as f:
                            with open('targets_' + args.section + '.txt', 'w') as h:
                                with open('speakers_' + args.section + '.txt', 'w') as s:
                                    # with open('features_'+args.section+'.txt', 'w') as m:
                                    with open('topics_' + args.section + '.txt', 'w') as n:
                                        with open(args.section + q_path, 'w') as d:
                                            for line, grd, conf_score in zip(g.readlines(), f.readlines(),
                                                                             conf_raw.readlines()):
                                                grd = grd.replace('\n', '')
                                                line = line.replace('\n', '')
                                                conf_line = conf_score.replace('\n', '')
                                                line, conf_line = zip(
                                                    *((wrd, conf) for wrd, conf in zip(line.split(), conf_line.split())
                                                      if not re.match('(%HESITATION%)|(\S*_%partial%)', wrd)))
                                                if line[2][0:2] == 'SA':
                                                    if line[2] in SA_LIST:
                                                        topic = ' '.join(line[1:3])
                                                    else:
                                                        topic = line[2]
                                                else:
                                                    topic = ' '.join(line[1:3])
                                                try:
                                                    if len(line[3:]) > 2 and len(conf_line[3:]) > 2:
                                                        text = ' '.join(line[3:])
                                                        conf_scores = ' '.join(conf_line[3:])
                                                        question = q_dict[a_dict[topic]]
                                                        # feats = feature_dict[line[0]]
                                                        d.write(question + '\n')
                                                        h.write(grd + '\n')
                                                        t.write(text + '\n')
                                                        conf.write(conf_scores + '\n')
                                                        s.write(line[0] + '\n')
                                                        # m.write(feats+'\n')
                                                        n.write(topic + '\n')
                                                except:
                                                    missing.append(topic)


if __name__ == '__main__':
    main()
