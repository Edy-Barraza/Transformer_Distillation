# Knowledge Distillation For Transformer Language Models


Distill the knowledge of Google's BERT transformer language model into a smaller transformer. A blog post on the topic can be found 
[here](https://edy-barraza.github.io/week12/).

Using this repository for knowledge distillation is a 5-stage processes outlined as such:
<ol type="I">
    <b>
        <li> Download Pretrained Model</li>
        <li> Extract Wikipedia </li>
        <li> Prepare Text For TensorFlow </li>
        <li> Extract Teacher Neural Network Outputs </li>
        <li> Distill Knowledge </li>
    </b>
</ol>

<h3> Download Pretrained Model </h3>
In order to distill knowledge from a large pretrained transformer model, we need to first download that model! Links to these models are available in Google's original [BERT release repository readme](https://github.com/google-research/bert/blob/master/README.md) . For the purpose of this readme, we will assume you have downloaded BERT-Base Uncased (12-layer, 768-hidden, 12-heads, 110M parameters )within this repository. 

<h3>II. Extract Wikipedia</h3>
To extract the Wikipedia Corpus, follow the instructions outlined in [this](https://github.com/Edy-Barraza/Transformer_Distillation/tree/master/extract_wikipedia_for_bert) part of the repository. 

<h3> III. Prepare Text For TensorFlow </h3>
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

<h3>IV. Extract Teacher Neural Network Outputs</h3>
One possibility for performing knowledge distillation is to pass an input to the student and teacher networks at the same time and using the outputs of the teacher for the student to learn from. However, considering that this will put a strain on our RAM and that we will be making multiple runs through each of over our data, it is more resource efficient to run through all of our data once and save the output of our teacher network with the inputs that were fed to it. This is accomplished by running extract_teacher_labels_truncated.py

```
python extract_teacher_labels_truncated.py --bert_config_file uncased_L-12_H-768_A-12/bert_config.json --data/record_intermed/wiki_intermed_0
```
extract_teacher_labels_truncated.py has the following arguments:

```
Args:
    bert_config_file (str) : The config json file corresponding to the pre-trained BERT model. This specifies the model architecture
    input_file (str) : Input TF example files (can be a glob or comma separated)

```
    
 


<h3>V. Distill Knowledge</h3>

