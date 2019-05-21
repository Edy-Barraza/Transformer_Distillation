# Knowledge Distillation For Transformer Language Models


Distill the knowledge of Google's BERT transformer language model into a smaller transformer. A blog post on the topic can be found 
[here](https://edy-barraza.github.io/week12/).

Using this repository for knowledge distillation is a 4-stage processes outlined as such:
<ol type="I">
    <b>
    <li> Extract Wikipedia </li>
    <li> Prepare Text For TensorFlow </li>
    <li> Extract Teacher Neural Network Outputs </li>
    <li> Distill Knowledge </li>
    </b>
</ol>

<h3>I. Extract Wikipedia</h3>
To extract the Wikipedia Corpus, follow the instructions outlined in [this](https://github.com/Edy-Barraza/Transformer_Distillation/tree/master/extract_wikipedia_for_bert) part of this repository. 

<h3> II. Prepare Text For TensorFlow </h3>
After extracting Wikipedia, you should have a txt file of ~12GB in size. To prepare this text for TensorFlow, we must turn it into a tfrecord file. tfrecord files allow us to work with a dataset when we can't load all of it onto RAM. As an intermediary step, we must first slit this file into smaller ones in order not to run into RAM or disk space problems later down the line. Thus we must run split_text.py

```
python split_text.py --read_file wikipedia.txt --split_number 20 --folder data/split_dir --name_base wiki_split
```
split_text.py has the following arguments:

```
Args:
    read_file (str) : the txt file that will be split
    split_number (int) : the number of smaller txt files that will be created
    folder (str) : the path where the split txt files will be placed
    name_base (str) : the base name of the split txt files. files will be named as such: base_name_N where N is a number
```
