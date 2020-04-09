# SMS-Spam-Classification
Using recurrent neural network to classify SMS messages as spam or legitimate. The following shows my contribution to a course project titled ["Machine_Learning_Approaches_to_Spam_Filtering_Problems"](https://kenneth-lee-ch.github.io/files/Machine_Learning_Approaches_to_Spam_Filtering_Problems.pdf).

## Introduction
Spam, a bulk of unsolicited messages sent by anonymous sources, has been a costly issue to human communication. Machine learning techniques have been shown promising to filter these messages as it adapts to the evolving characteristics of the spam. In this work, we focus on neural networks to the problem of spam filtering. Overall, we conclude that using bidirectioncomparing various classification methodsal gated recurrent neural network with tokenizer method is the most robust way we have found to handle this problem with our particular dataset. This work provides insights on how to design a neural network to work with spam filtering problem.

## Methodology

### Data Preprocessing

We have found that the data has a skewed distribution, meaning that we have only 13.41 percent of the data that is labeled as spam and the rest is labeled as non-spam, namely ham (see `Figure 1`).

![](/figures/fig1.png)

*Figure 1: Distribution of the labels*


Next, we use the Natural Language Toolkit (nltk) package in Python to clean the data as follows:
 
#### Lemmetization 
First, we lemmetize the data to ensure all the related forms of a word to have a common base form.

#### Stopwords and punctuation Removal

Then, after we removed the punctuation of each word, we take away the stopwords in the messages by using the default English stopwords list in nltk. Afterwards, we turn every word into lowercase.

The word clouds and the frequency plots for the spam and ham texts are shown in the figure \ref{wordcloud}.

#### Splitting Training and Testing datsets

Next, we splitted the dataset into training and testing set by having 15 percent of the data being testing and 85 percent being training.

#### Vectorization

In this project, it involves three different ways to tokenize both training and testing data: CountVectorizer, Term Frequency-Inverse Document Frequency(TF-IDF) Vectorizer, and Tokenization. 

### Evaluation Metrics

Since spam filtering is a binary classification problem and the dataset has an imbalanced distribution for two output classes, precision and recall will be a more informative metric than the Receiver Operating Characteristic (ROC) curve to evaluate the classifiers we use in this work.

### Modeling algorithms

#### Bidirectional Gated Recurrent Unit (GRU) with embedding from scratch

There are five main components of this neural network architectures: word embedding input layer, GRU, bidirectional layer, global max pooling layer, and the dropout layer. The output layer is a sigmoid activation function. First, the input size of the embedding layer is the number of unique words plus one by the number of dimension we would like to use to represent each word vector, which we set to 100 in our case. We also set the batch size to be 512 and use Adam optimizer with a binary cross-entropy loss function. Besides the main five mentioned previously, we set the activation in our hidden layer to be ReLU. The thresold value for the dropout layer is 0.1 to slightly penalize the model.

#### Bidirectional Gated Recurrent Unit (GRU) with pre-trained embedding layer

Instead of training a word embedding layer from sratch, we use a pre-trained word embedding layer called Global Vectors for Word Representation (GloVe). GloVe was trained on a dataset that contains 6 billion tokens with 400,000 vocabularies. There are several downloadable options; we picked the smallest package of embeddings that has 300 dimensions. The set up is the same as the GRU with wording embedding from scratch, except for the embedding layer, which will be replaced by the pre-trained word embedding layer.

## Results

When using GRU with embedding from scratch, we can achieve F1-score of 0.98 with 3 layers and 36 nodes. Then, we see that using GRU with pre-trained word embedding layer has not improved the F1-score. We can see that comparison from a preicison-and-recall curve as shown by `Figure 2`.

![](/figures/fig2.png)

*Figure 2: Precision-and-recall curve to compare three different setups (36 nodes with 1/3/6 layers) for bidirectional Gated Recurrent Unit (GRU) with embedding from scratch*

## Conclusion and discussion

### Pre-tained Layer is not always better

In the RNN architecture, we attempted to apply pre-trained word embedding layer to improve the model, but the result shows that it isn't always a better choice. 

### Overfitting
Also, we see that from RNN architecture, the F1-score doesn't increase simply by increasing the number of nodes and layers in the model. The RNN may have suffered from overfitting issues. The structure of the layers will need further investigation. There maybe potential accuracy gain by adding more dropout layers in between the hidden layers to better regularize the model. 

### Vectorization methods matter

Moreover, Bidirectional GRU is able to achieve the highest F1 score based on tokenizer vectorization approach. Potentially, this gain may come from the fact that bidirectional GRU takes the order of the text into account when it "learns" how to classify the spam messages. In addition, we see that not all the models work well with the same vectorization method. For example, using CountVector and TD-IDF has significantly slowed down the training time of recurrent neural network due to the sparse matrices generated by the vectorizers. 

### Hyperparameters Tuning
Tuning hyperparameters is a time-consuming and exhaustive task to know when the model should perform the best. 

## Data Source
[SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset). Note that you must download the pre-trained word vector "glove6B.zip" from here:https://nlp.stanford.edu/projects/glove/ in order to run the code about gated recurrent unit with pretrained word embedding layer. You need to unzip the folder once you have downloaded it. Then put "glove.6B.300d.txt" on your working directory.

## Reference

Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results.  Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011.

Gómez Hidalgo, J.M., Almeida, T.A., Yamakami, A. On the Validity of a New SMS Spam Collection.  Proceedings of the 11th IEEE International Conference on Machine Learning and Applications (ICMLA'12), Boca Raton, FL, USA, 2012.

Almeida, T.A., Gómez Hidalgo, J.M., Silva, T.P.  Towards SMS Spam Filtering: Results under a New Dataset.   International Journal of Information Security Science (IJISS), 2(1), 1-18, 2013. 

Schuster, M. and Paliwal, K. K. Bidirectional recurrent neu-ral networks.IEEE Transactions on Signal Processing,45(11):2673–2681, 1997

Saito, T. and Rehmsmeier, M.   The precision-recall plotis more informative than the roc plot when evaluatingbinary classifiers on imbalanced datasets.PloS one, 10(3):e0118432, 2015

Ramos, J. et al. Using tf-idf to determine word relevance indocument queries. InProceedings of the first instructionalconference on machine learning, volume 242, pp. 133–142. Piscataway, NJ, 2003.

Jozefowicz, R., Zaremba, W., and Sutskever, I. An empiricalexploration of recurrent network architectures. InInterna-tional Conference on Machine Learning, pp. 2342–2350,2015

Davis,  J.  and  Goadrich,  M.    The  relationship  betweenprecision-recall and roc curves.  InProceedings of the23rd international conference on Machine learning, pp.233–240. ACM, 2006.

Cho,  K.,  Van Merri ̈enboer,  B.,  Gulcehre,  C.,  Bahdanau,D., Bougares, F., Schwenk, H., and Bengio, Y.  Learn-ing  phrase  representations  using  rnn  encoder-decoderfor  statistical  machine  translation.arXiv  preprintarXiv:1406.1078, 2014.

Bengio, Y., Simard, P., Frasconi, P., et al.  Learning long-term dependencies with gradient descent is difficult.IEEEtransactions on neural networks, 5(2):157–166, 1994.

Agarap, A. F.   Deep learning using rectified linear units(relu).arXiv preprint arXiv:1803.08375, 2018