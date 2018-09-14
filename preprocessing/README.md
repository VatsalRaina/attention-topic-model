# File preprocessing
Generation of a new training dataset is done in two steps:
### Process the transcriptions and save the processed files as readable .txt files:
```
python magic_bruno_script_of_pywer.py /home/alta/BLTSpeaking/convert-v2/4/lib/script.v7/scripts.mlf /home/alta/BLTSpeaking/exp-ar527/grammar/detection/data/word/mlf/BLXXXeval1.mlf /home/miproj/urop.2018/bkm28/test_dataset_preprocessing/ --fixed_sections A B --exclude_sections A B --multi_sections E --multi_sections_master SE0006 --fixed_questions --speaker_grades_path /home/alta/BLTSpeaking/convert-v2/4/lib/grades-orig/BLXXXeval1.lst --exclude_grades 1
```
### Generate the required tfrecords files from the processed .txt files
```
python magic_bruno_script_of_virTFue.py /home/miproj/urop.2018/bkm28/test_dataset_preprocessing /home/alta/BLTSpeaking/exp-am969/relevance-experiments/data/input.wlist.index /home/miproj/urop.2018/bkm28/test_dataset_preprocessing/tfrecords --valid_fraction 0.1 --preprocessing_type train
```



## Relevant Paths:
#### Scripts (prompts):
###### BULATS
`/home/alta/BLTSpeaking/convert-v2/4/lib/script.v7/scripts.mlf`
###### LINSK
`/home/alta/Linguaskills/convert/lib/script/scripts.mlf`

#### Responses:
###### BLXXXgrp24
`/home/alta/BLTSpeaking/exp-ar527/grammar/detection/data/word/mlf/BLXXXgrd00_grd01_grd02_uns00_uns10_uns11.mlf`

###### BLXXXeval{1,3}
`/home/alta/BLTSpeaking/exp-ar527/grammar/detection/data/word/mlf/BLXXX{dataset}.mlf`

###### LINSKuns03
`/home/alta/Linguaskills/exp-graphemic-kmk/D3/align-LINSKuns03.mlp.mpe/lib/wlabs/train.mlf`

###### LINSKevl03
`/home/alta/Linguaskills/exp-graphemic-kmk/D3/align-LINSKevl03.mlp.mpe/lib/wlabs/train.mlf`

#### Grades:
###### BLXXXeval{1,3} and BLXXXgrp24
  `/home/alta/BLTSpeaking/convert-v2/4/lib/grades-orig/ BLXXX{dataset}.lst`
  
###### LINSK
  `/home/alta/Linguaskills/convert/lib/grades-orig/LINSK{dataset}.lst`
  
  

