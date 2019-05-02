import argparse
from gensim.corpora.wikicorpus import WikiCorpus
from nltk.tokenize import sent_tokenize

# noinspection PyPep8
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        type=str,
                        required=True,
                        default="enwiki-latest-pages-articles.xml.bz2",
                        help="(str) link to wikipedia dump")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="(str) the name of the output txt file you wish to write to. Will create new file with given name if the file doesn't exist ")
    args = parser.parse_args()

    generate_txtcorpus(args.input_file,args.output_file)

def generate_txtcorpus(input_file,output_file):
    output_f = open(output_file,'w')
    wiki_corpus = WikiCorpus(input_file,lemmatize=False,article_min_tokens=1,token_min_len=1,token_max_len=100)
    i=0
    #returns the text, (page_id, title)
    for text,(_,_) in wiki_corpus.get_texts():
        print("from get_text(): ")
        print(text)
        print("joined text:")
        print(" ".join(text))

        text = " ".join(text)
        sentences = sent_tokenize(text)
        for sentence in sentences:
            output_f.write(bytes(sentence,'utf-8'))
            #output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        output_f.close()
        break









