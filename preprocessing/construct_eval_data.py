#! /usr/bin/env python
import numpy as np
from math import floor
import sys, os, re
import argparse

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('datafile', type=str,
                               help='Absolute path to data file')
commandLineParser.add_argument('conffile', type=str,
                               help='Absolute path to conf score file')
commandLineParser.add_argument('questionfile', type=str,
                               help='Absolute path to question file')
commandLineParser.add_argument('gradefile', type=str,
                               help='Absolute path to target file')
commandLineParser.add_argument('speakerfile', type=str,
                               help='Absolute path to speaker file')
commandLineParser.add_argument('bulats_gradefile', type=str,
                               help='Absolute path to speaker file')
commandLineParser.add_argument('name', type=str,
                               help='Name of dataset to create')
commandLineParser.add_argument('path', type=str,
                               help='Name of dataset to create')
commandLineParser.add_argument ('section', type=str,
                                help = 'Absolute path to feature file')
commandLineParser.add_argument('--samples', type=int, default=10,
                               help='Name of dataset to create')
# commandLineParser.add_argument ('featurefile', type=str,
#                                help = 'Absolute path to feature file')

def main(argv=None):
    args = commandLineParser.parse_args()

    # Open All the files
    with open(args.datafile, 'r') as d:
        data = []
        for line in d.readlines():
            data.append(line)
    with open(args.conffile, 'r') as d:
        confs = []
        for line in d.readlines():
            confs.append(line)
    with open(args.questionfile, 'r') as q:
        questions = []
        for line in q.readlines():
            questions.append(line)
    with open(args.gradefile, 'r') as t:
        grades = []
        for line in t.readlines():
            grades.append(float(line.replace('\n', '')))

    with open(args.speakerfile, 'r') as s:
        speakers = []
        for line in s.readlines():
            speakers.append(line)
    with open(args.bulats_gradefile, 'r') as f:
        bulats = []
        for line in f.readlines():
            bulats.append(float(line.replace('\n', '')))

    #  with open(args.featurefile, 'r') as f:
    #    features=[]
    #    for line in f.readlines():
    #      features.append(line)

    # Copy questions
    np.random.seed(1000)
    shuf_questions = questions[:]
    shuf_questions = np.random.permutation(shuf_questions)
    pre_A1 = []
    A1 = []
    A2 = []
    B1 = []
    B2 = []
    C1 = []
    C2 = []

    rel = 0
    tot = 0
    for sample in xrange(args.samples):
        for dat, conf, ques, sques, tar, spkr, bul in zip(data, confs, questions, shuf_questions, grades, speakers,
                                                          bulats):
            rel+=1
            tot += 2
            if ques == sques:
                qtar = 1
                rel += 1
            else:
                qtar = 0

            if int(np.floor(tar)) == 0:
                pre_A1.append([dat, conf, ques, 1, spkr, bul])
                pre_A1.append([dat, conf, sques, qtar, spkr, bul])
            elif int(np.floor(tar)) == 1:
                A1.append([dat, conf, ques, 1, spkr, bul])
                A1.append([dat, conf, sques, qtar, spkr, bul])
            elif int(np.floor(tar)) == 2:
                A2.append([dat, conf, ques, 1, spkr, bul])
                A2.append([dat, conf, sques, qtar, spkr, bul])
            elif int(np.floor(tar)) == 3:
                B1.append([dat, conf, ques, 1, spkr, bul])
                B1.append([dat, conf, sques, qtar, spkr, bul])
            elif int(np.floor(tar)) == 4:
                B2.append([dat, conf, ques, 1, spkr, bul])
                B2.append([dat, conf, sques, qtar, spkr, bul])
            elif int(np.floor(tar)) == 5:
                C1.append([dat, conf, ques, 1, spkr, bul])
                C1.append([dat, conf, sques, qtar, spkr, bul])
            elif int(np.floor(tar)) == 6:
                C2.append([dat, conf, ques, 1, spkr, bul])
                C2.append([dat, conf, sques, qtar, spkr, bul])
            shuf_questions = np.random.permutation(shuf_questions)


    print 'percent relevant:', float(rel) / float(tot)
    print len(pre_A1), len(A1), len(A2), len(B1), len(B2), len(C1), len(C2)
    data = []
    data.extend(pre_A1)
    data.extend(A1)
    data.extend(A2)
    data.extend(B1)
    data.extend(B2)
    data.extend(C1)
    data.extend(C2)
    data = np.random.permutation(data)

    print len(bulats)
    with open(os.path.join(args.path, args.name + '_' + args.gradefile.split('/')[-1]), 'w') as g:
        with open(os.path.join(args.path, args.name + '_' + args.conffile.split('/')[-1]), 'w') as c:
            with open(os.path.join(args.path, args.name + '_' + args.datafile.split('/')[-1]), 'w') as d:
                with open(os.path.join(args.path, args.name + '_' + args.questionfile.split('/')[-1]), 'w') as q:
                    with open(os.path.join(args.path, args.name + '_targets_'+args.section+'.txt'), 'w') as t:
                        with open(os.path.join(args.path, args.name + '_' + args.speakerfile.split('/')[-1]), 'w') as s:
                        #   with open(name+'_topic_'+args.featurefile, 'w') as f:
                            for dat, conf, ques, tar, spkr, grd in data:
                                d.write(dat)
                                q.write(ques)
                                t.write(str(tar) + '\n')
                                s.write(spkr)
                                c.write(conf)
                                # f.write(feat)
                                g.write(str(int(floor(float(grd)))) + '\n')


if __name__ == '__main__':
    main()
