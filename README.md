# Knowledge Distillation For Transformer Language Models


Distill the knowledge of Google's BERT transformer language model into a smaller transformer. A blog post on the topic can be found 
[here](https://edy-barraza.github.io/week12/).

Using this repository for knowledge distillation is a 5-stage processes with a couple options to do distributed training. Here is an outline:
<ol type="I">
    <b>
        <li> Download Pretrained Model</li>
        <li> Extract Wikipedia </li>
        <li> Prepare Text For TensorFlow </li>
        <li> Extract Teacher Neural Network Outputs </li>
        <li> Distill Knowledge </li>
        <li> Single-Node Distributed Distillation </li>
        <li> Multi-Node Distributed Distillation </li>
    </b>
</ol>

<h3> Download Pretrained Model </h3>

In order to distill knowledge from a large pretrained transformer model, we need to first download that model! Links to these models are available in Google's original [BERT release repository readme](https://github.com/google-research/bert/blob/master/README.md). For the purpose of this readme, we will assume you have downloaded BERT-Base Uncased (12-layer, 768-hidden, 12-heads, 110M parameters ) within this repository. 

<h3>II. Extract Wikipedia</h3>

We will use Wikipedia in our training data. To extract the Wikipedia Corpus, follow the instructions outlined in [this](https://github.com/Edy-Barraza/Transformer_Distillation/tree/master/extract_wikipedia_for_bert) part of the repository. This will ultimately produce a directory of many managable txt files containing Wikipedia!

<h3> III. Prepare Text For TensorFlow </h3>
We can turn our Wikipedia txt files into tfrecord files with masked tokens by running create_pretraining_data.py

```
python create_pretraining_data.py \
    --input_dir data/split_dir/ \
    --output_dir data/record_intermed \
    --output_base_name wiki_intermed \
    --vocab_file uncased_L-12_H-768_A-12/vocab.txt
```
create_pretraining_data.py has the following arguments:

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

One possibility for performing knowledge distillation is to pass an input to the student and teacher networks at the same time and using the outputs of the teacher for the student to learn from. However, considering that this will put a strain on our RAM and that we will be making multiple runs through each of over our data, it is more resource efficient to run through all of our data once and save the output of our teacher network with the inputs that were fed to it. This is accomplished by running produce_teacher_labels.py . The teacher labels are BERT's predicted softmax distribution over it's vocabulary for any given masked token. Given that BERT's vocabulary is ~30,000 in size, I experimented with truncating with the top K probabilities, which proved to degrade performance. Alas, I will keep this functionality with the `truncation_factor` argument in hopes that it will be useful to someone one day. 

```
python produce_teacher_labels.py \
    --bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
    --data/record_intermed/wiki_intermed_0.tfrecord \
    --output_file data/record_distill/wiki_distill_0.tfrecord \
    --truncation_factor 10 \
    --init_checkpoint uncased_L-12_H-768_A-12/bert_model.ckpt 
```
extract_teacher_labels_truncated.py has the following arguments:

```
Args:
    bert_config_file (str) : The config json file corresponding to the pre-trained BERT model. This specifies the model architecture
    input_file (str) : Input TF example files (can be a glob or comma separated)
    output_file (str) : The output file that has transformer inputs and teacher outputs
    truncation_factor (int) : Number of top probable words to save from teacher network output. If `0`, the whole softmax distribution is saved
    init_checkpoint (str) : Initial checkpoint (usually from a pre-trained BERT model)
    max_seq_length (int) : The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded. Must match data generation
    max_predictions_per_seq (int) : Maximum number of masked LM predictions per sequence. Must match data generation
    batch_size (int) : Total batch size when processing sequences
```

<h3>V. Distill Knowledge</h3>

Now that we have our teacher outputs we can start training a student network! To run on a single machine run network_distillation_single_machine_truncated.py 

```
python network_distillation_single_machine_truncated.py \
    --bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
    --input_file data/record_distill/wiki_distill_0.tfrecord \
    --output_dir output_dir \
    --truncation_factor 10 \
    --do_train True \
    --do_eval true
```

network_distillation_single_machine_truncated.py has the following arguments:

```
Args:
    bert_config_file (str) : The config json file corresponding to the pre-trained BERT model. This specifies the model architecture
    input_file (str) : Input TF example files (can be a glob or comma separated)
    output_dir (str) : The output directory where the model checkpoints will be written
    init_checkpoint (str) : Initial checkpoint (usually from a pre-trained BERT model)
    truncation_factor (int) : Number of top probable words to save from teacher network output
    do_train (bool) : Whether to run training
    do_eval (bool) : Whether to run eval on the dev set
    max_seq_length (int) : The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded. Must match data generation
    max_predictions_per_seq (int) : Maximum number of masked LM predictions per sequence. Must match data generation
    train_batch_size (int) : Total batch size for training
    eval_batch_size (int) Total batch size for eval
    learning_rate (float) : The initial learning rate for Adam
    num_train_steps (int) : Number of training steps
    num_warmup_steps (int) Number of warmup steps
    save_checkpoints_steps (int) : How often to save the model checkpoint
    iterations_per_loop (int) : How many steps to make in each estimator call
    max_eval_steps (int) : Maximum number of eval steps
```

<h3>VI. Single-Node Distributed Distillation </h3>

Now suppose you have a lil cluster of 8 GPU's! If you have Horovod installed, you can perform some distributed training!!! (If you don't have horovod installed you can install it [here](https://github.com/horovod/horovod#install)). We shall run network_distillation_distributed_truncated.py to perform distributed training as such:

```
mpirun -np 8 \
    -H localhost:8 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python network_distillation_distributed_truncated.py \
    --bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
    --input_file data/record_distill/wiki_distill_0.tfrecord \
    --output_dir output_dir \
    --truncation_factor 10 \
    --do_train True \
    --do_eval true
```

network_distillation_distributed_truncated.py has the following arguments:


```
Args:
    bert_config_file (str) : The config json file corresponding to the pre-trained BERT model. This specifies the model architecture
    input_file (str) : Input TF example files (can be a glob or comma separated)
    output_dir (str) : The output directory where the model checkpoints will be written
    init_checkpoint (str) : Initial checkpoint (usually from a pre-trained BERT model)
    truncation_factor (int) : Number of top probable words to save from teacher network output
    do_train (bool) : Whether to run training
    do_eval (bool) : Whether to run eval on the dev set
    max_seq_length (int) : The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded. Must match data generation
    max_predictions_per_seq (int) : Maximum number of masked LM predictions per sequence. Must match data generation
    train_batch_size (int) : Total batch size for training
    eval_batch_size (int) Total batch size for eval
    learning_rate (float) : The initial learning rate for Adam
    num_train_steps (int) : Number of training steps
    num_warmup_steps (int) Number of warmup steps
    save_checkpoints_steps (int) : How often to save the model checkpoint
    iterations_per_loop (int) : How many steps to make in each estimator call
    max_eval_steps (int) : Maximum number of eval steps
```

<h3>VII. Multi-Node Distributed Distillation </h3>

Suppose you have a boatload of credits on AWS... Now I can explain how to run distributed training on multiple EC2 instances, each acting as a node with 8 GPU's. This will get a lil bit more hairy than our previous training procedures, but will also be more rewarding :)

Here is a summary of the process:
<ol type="i">
  <b>
  <li>Create IAM Role</li>
  <li>Create Security Group</li>
  <li>Initialize Leader</li>
  <li>Launch Workers</li>
  <li>Finalize Leader</li>
    </b>
</ol>  

<h4> i. Create IAM Role </h4>

First, we will create an IAM Role. We need our multiple EC2 instances to be able to communicate with each other via SSH. Instead of enabling this by managing our instances' SSH keys, we can get AWS to do that for us! We will create a keypair on the leader instance of our cluster, and use the Simple Systems Manager (SSM) service to store the public part of SSH key and retrieve it from the worker nodes during the launch process. To create our IAM Role or edit one you're already using, we will go to [IAM in our Services on the AWS console](https://console.aws.amazon.com/iam/home?region=us-east-1#/home).

If we're creating a new role, click on 'Roles' and then 'Create Role'. For 'Select type of trusted entity', choose 'AWS service'. At 'Choose the service that will use this role', choose 'EC2'. Move onto 'Next: Permissions'. At 'Filter policies', choose "AmazonSSMFullAccess'. Hit 'Next:Tags', where here you can put any tags if you feel like it. Hit 'Next:Review', and at 'Role name' put something you can remember for later. I'm gonna use 'SSMFullAccessRole'. Now hit 'Create role'.

<h4>ii. Create Security Group </h4>

Sweet now we can create our security group! Navigate to the [EC2 service on the AWS console](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1). Under 'Resources' select 'Security Groups'. Then click 'Create Security Group'. I will name mine 'TensorFlow', but you can name yours whatever tickles your fancy. Add an inbound rule, allowing traffic on port 22 TCP for just your IP address. Create the security group, and take note of the Group ID (sg-...). Then edit the security group by adding an inbound rule allowing all traffic from the security group as a source. 

<h4> iii. Initialize Leader </h4>

Now we will launch our leader instance using the Ubuntu Deep Learning AMI 22.0 with our newly created Security Group and IAM Role. To ensure access, we SSH into this instance as such:

```
ssh-add -K your_key.pem
ssh -A ubuntu@public_ip_of_leader
```

Once we are connected to the leader through SSH, we run the following commands:

```
ssh-keygen -q -N "" -t rsa -b 4096 -f /home/ubuntu/.ssh/id_rsa ; cat /home/ubuntu/.ssh/id_rsa.pub >> /home/ubuntu/.ssh/authorized_keys ; aws --region us-east-1 ssm put-parameter --name 'TensorFlowClusterPublicKey' --type=SecureString --value=file:///home/ubuntu/.ssh/id_rsa.pub
```

These commands generate an SSH key pair locally and also upload it to the SSM service so we can use it later on the worker instances. To verify that the key was correctly uploaded to the SSM service you can check [here](https://console.aws.amazon.com/systems-manager/parameters/TensorFlowClusterPublicKey/description?region=us-east-1)

<h4> iv. Launch Workers</h4>

Now we will launch our worker instances! You can select an arbitrary number of worrker instances for distributed training, just  make sure to use the same AMI, Security Group, Availability, and IAM Role as previously mentioned. In addition, we must also go to the 'Advanced Details' portion of the 'Configure Instance' page when launching the instances. Here in the 'User data' text box we must include the following command:

```
#!/bin/bash
sleep 10s ; aws --region us-east-1 ssm get-parameter --name 'TensorFlowClusterPublicKey' --with-decryption --query 'Parameter.Value' --output text >> /home/ubuntu/.ssh/authorized_keys
```

This command will be executed automatically when the worker instances are launched. It will get the public part of your SSH key from the SSM service and apply it accordingly to each worker instance. 

<h4> v. Finalize Leader </h4>

Now that we have launched our worker instances, we can go back to our leader instance to control our distributed training. We have a couple of ways of getting all of our files on all of our instances depending on whether we have them on the leader or not. 

If we already have all of our files on the leader instance. Add to the hosts file `hosts` the private IPs of all our instances, and the number of GPU's each instance has. The file will look as such:
```
172.100.1.200 slots=8
172.200.8.99 slots=8
172.48.3.124 slots=8
localhost slots=8
```
Then run the following commands:

```
function copyclust(){ while read -u 10 host; do host=${host%% slots*}; rsync -azv "$2" $host:"$3"; done 10<$1; };
copyclust hosts ~/Transformer_Distillation ~/Transformer_Distillation
```

If we don't have our files on any instance, but have them all on S3, you will first create a hosts file with the the private IPs of all our instances and the number of GPU's each instance has as previously described.

Then run the following command:
```
function runclust(){ while read -u 10 host; do host=${host%% slots*}; if [ ""$3"" == "verbose" ]; then echo "On $host"; fi; ssh -o "StrictHostKeyChecking no" $host ""$2""; done 10<$1; };
runclust hosts "tmux new-session -d \"aws s3 sync s3://your-transformer-bucket ~/Transformer_Distillation/ \""
mv hosts ~/Transformer_Distillation/hosts
```

Now that we have our files on all of our instances, here's our last step before training!!! Run the following commands:
```
source activate tensorflow_p36
function runclust(){ while read -u 10 host; do host=${host%% slots*}; if [ ""$3"" == "verbose" ]; then echo "On $host"; fi; ssh -o "StrictHostKeyChecking no" $host ""$2""; done 10<$1; };
runclust hosts "echo \"StrictHostKeyChecking no\" >> ~/.ssh/config"
```

Now we can train in peace, running 
```
chmod +x multinode_train.sh
multinode_train.sh TOTAL_GPUS
```

where TOTAL_GPUS is the total number of GPUS on all of our instances. Hope you enjoyed :) 

