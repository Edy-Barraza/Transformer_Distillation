import argparse
import os
import subprocess as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="(str) Absolute path to the directory containing the txt files that will be converted to tfrecords")

    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="(str) Absolute path to the Output TF example file, or comma-separated list of files")

    parser.add_argument("--vocab_file",
                        type=str,
                        required=True,
                        help="Absolute path to the vocabulary file that the BERT model was trained on")

    parser.add_argument("--create_pretraining_subdir",
                        type=str,
                        default="bert/",
                        help="(str) Absolute path to the directory containing create_pretraining_data.py")

    parser.add_argument("--do_lower_case",
                        type=str,
                        default="True",
                        help="Whether to lower case the input text. Should be True for uncased models and False for cased models")

    parser.add_argument("--max_seq_length",
                        type=int,
                        default=128,
                        help="(int) Maximum sequence length")

    parser.add_argument("--max_predictions_per_seq",
                        type=int,
                        default=20,
                        help="(int) Maximum number of masked LM predictions per sequence")

    parser.add_argument("--random_seed",
                        type=int,
                        default=12345,
                        help="(int) Random seed for data generation")

    parser.add_argument("--dupe_factor",
                        type=int,
                        default=10,
                        help="(int) Number of times to duplicate the input data (with different masks)")

    parser.add_argument("--masked_lm_prob",
                        type=float,
                        default=.15,
                        help="(float) Masked LM probability")

    parser.add_argument("--short_seq_prob",
                        type=float,
                        default=.1,
                        help="(float) Probability of creating sequences which are shorter than the maximum length")

    args = parser.parse_args()

    if args.create_pretraining_subdir[-1] != "/":
        args.create_pretraining_subdir = args.reate_pretraining_subdir + "/"

    if args.input_dir[-1] != "/":
        args.input_dir = args.input_dir + "/"

    if args.do_lower_case != "True":
        if args.do_lower_case != "False":
            print("do_lower_case must be True or False")
            exit()

    for file in os.listdir(args.input_dir):
        sp.call("python create_pretraining_data.py " + \
                " --output_file "+args.output_file + \
                " --vocab_file "+args.vocab_file+ \
                " --do_lower_case " + args.do_lower_case + \
                " --max_seq_length " + str(args.max_seq_length) + \
                " --max_predictions_per_seq "+ str(args.max_predictions_per_seq) + \
                " --random_seed " + str(args.random_seed) + \
                " --dupe_factor " + str(args.dupe_factor) + \
                " --masked_lm_prob " + str(args.masked_lm_prob) + \
                " --short_seq_prob " + str(args.short_seq_prob) + \
                " --input_file " + args.input_dir + " " + file,
                shell=True,
                cwd=args.create_pretraining_subdir)




if __name__ == "__main__":
    main()

