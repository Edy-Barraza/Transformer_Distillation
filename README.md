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

After splitting Wikipedia into smaller txt files, we can turn all of them into tfrecord files by running multifile_create_pretraining_data.py

```
python multifile_create_pretraining_data.py --input_dir data/split_dir/ --output_dir data/record_intermed --output_base_name wiki_intermed --vocab_file uncased_L-12_H-768_A-12/vocab.txt
```
multifile_create_pretraining_data.py has the following arguments:

```
Args:
    input_dir (str) : Input directory of raw text files
    output_dir (str) : Output directory for created tfrecord files
    output_base_name (str) : Output base name for TF example files
    vocab_file (str) : The vocabulary file that the BERT model was trained on
    do_lower_case (bool) : Whether to lower case the input text. Should be True for uncased models and False for cased models
    max_seq_length (int) : Maximum sequence length
    max_predictions_per_seq (int) : Maximum number of masked LM predictions per sequence
    random_seed (int) : Random seed for data generation
    dupe_factor (int) : Number of times to duplicate the input data (with different masks)
    masked_lm_prob (float) : Masked LM probability
    short_seq_prob (float) : Probability of creating sequences which are shorter than the maximum length
```
