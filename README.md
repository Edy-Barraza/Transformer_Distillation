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

<h3>Extract Wikipedia</h3>
To extract the Wikipedia Corpus, follow the instructions outlined in [this](https://github.com/Edy-Barraza/Transformer_Distillation/tree/master/extract_wikipedia_for_bert) part of this repository. 
