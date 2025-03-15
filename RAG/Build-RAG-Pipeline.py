'''load openai-key'''
from dotenv import load_dotenv, find_dotenv
load_dotenv('.env')

'''
Loading Data (Ingestion)
Before your chosen LLM can act on your data, you first need to process the data and load it.
'''

# Loading using SimpleDirectoryReader, this is the easiest way to load data from directory
# which creates documents out of every file in a given directory.

# 1.Load data
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("data").load_data()
print(documents)

text = ["""History of NLP
The conception of natural language processing dates to the 1950s. In
1950, Alan Turing published an article titled “Computing Machinery
and Intelligence,” which discussed a method to determine whether a
machine exhibits human-like intelligence. This proposed test, most
popularly referred to as the Turing test, is widely considered as what
inspired early NLP researchers to attempt natural language
understanding.
The Turing test involves a setup where a human evaluator interacts
with both a human and a machine without knowing which is which.
The evaluator’s task is to determine which participant is the machine
and which is the human based solely on their responses to questions or
prompts. If the machine is successful in convincing the evaluator that it
is human, then it is said to have passed the Turing test. The Turing test
thus provided a concrete and measurable goal for AI research. Turing’s
proposal sparked interest and discussions about the possibility of
creating intelligent machines that could understand and communicate
in natural language like humans. This led to the establishment of NLP
as a fundamental research area within AI.
In 1956, with the establishment of the artiicial intelligence
research ield, NLP became an established ield of research in AI,
making it one of the oldest subields in AI research.
During the 1960s and 1970s, NLP research predominantly relied on
rule-based systems. One of the earliest NLP programs was the ELIZA
chatbot, developed by Joseph Weizenbaum between 1964 and 1966.
ELIZA used pattern matching and simple rules to simulate conversation
between the user and a psychotherapist. With an extremely limited
vocabulary and ruleset ELIZA was still able to articulate human-like
interactions. The General Problem Solver (GPS) system, developed in
the 1970s by Allen Newell and Herbert A. Simon, working with means-
end analysis, also demonstrated some language processing capabilities.
In the 1970s and 1980s, NLP research began to incorporate
linguistic theories and principles to understand language better. Noam
Chomsky’s theories on generative grammar and transformational
grammar inluenced early NLP work. These approaches aimed to use
linguistic knowledge and formal grammatical rules to understand and
process human language.
The following are some key aspects of linguistic-based approaches
in NLP.
Formal Grammars
Linguistics-based NLP heavily relied on formal grammars, such as
context-free grammars and phrase structure grammars. These
formalisms provided a way to represent the hierarchical structure and
rules of natural language sentences.
Transformational Grammar and Generative Grammar
Noam Chomsky’s transformational grammar and generative grammar
theories signiicantly inluenced early NLP research. These theories
focused on the idea that sentences in a language are generated from
underlying abstract structures, and rules of transformation govern the
relationship between these structures.
Parsing and Syntactic Analysis
Parsing, also known as syntactic analysis, was a crucial aspect of
linguistics-based NLP. It involved breaking down sentences into their
grammatical components and determining the hierarchical structure.
Researchers explored various parsing algorithms to analyze the syntax
of sentences.
Context and Semantics
Linguistics-based approaches aimed to understand the context and
semantics of sentences beyond just their surface structure. The focus
was on representing the meaning of words and phrases in a way that
allowed systems to reason about their semantic relationships.
Language Understanding
Linguistics-based NLP systems attempted to achieve deeper language
understanding by incorporating syntactic and semantic knowledge.
This understanding was crucial for more advanced NLP tasks, such as
question answering and natural language understanding.
Knowledge Engineering
In many cases, these approaches required manual knowledge
engineering, where linguistic rules and structures had to be explicitly
deined by human experts. This process was time-consuming and
limited the scalability of NLP systems.
There are, however, some limitations in linguistics-based NLP
approaches. While linguistics-based approaches had theoretical appeal
and offered some insights into language structure, they also faced
limitations. The complexity of natural languages and the vast number of
exceptions to linguistic rules made it challenging to develop
comprehensive and robust NLP systems solely based on formal
grammars.
Because of these limitations, while linguistic theories continued to
play a role in shaping the NLP ield, they were eventually
complemented and, in some cases, surpassed by data-driven
approaches and statistical methods.
During the 1990s and 2000s, NLP started shifting its focus from
rule-based and linguistics-driven systems to data-driven methods.
These approaches leveraged large amounts of language data to build
probabilistic models, leading to signiicant advancements in various
NLP tasks.
Statistical NLP methods used several approaches and applications.
Let us look at a few next.
Probabilistic Models
Statistical approaches relied on probabilistic models to process and
analyze language data. These models assigned probabilities to different
linguistic phenomena based on their occurrences in large annotated
corpora.
Hidden Markov Models
Hidden Markov models (HMMs) were one of the early statistical models
used in NLP. They were employed for tasks such as part-of-speech
tagging and speech recognition. HMMs use probability distributions to
model the transition between hidden states, which represent the
underlying linguistic structures.
N-Gram Language Models
N-gram language models became popular during this era. They
predicted the likelihood of a word occurring given the preceding (n-1)
words. N-grams are simple but effective for tasks such as language
modelling, machine translation, and information retrieval.
Maximum Entropy Models
Maximum entropy (MaxEnt) models were widely used in various NLP
tasks. They are a lexible probabilistic framework that can incorporate
multiple features and constraints to make predictions.
Conditional Random Fields
Conditional random ields (CRFs) gained popularity for sequence
labeling tasks, such as part-of-speech tagging and named entity
recognition. CRFs model the conditional probabilities of labels given the
input features.
Large Annotated Corpora
Statistical approaches relied on large annotated corpora for training
and evaluation. These corpora were essential for estimating the
probabilities used in probabilistic models and for evaluating the
performance of NLP systems.
Word Sense Disambiguation
Statistical methods were applied to word sense disambiguation (WSD)
tasks, where the goal was to determine the correct sense of a
polysemous word based on context. Supervised and unsupervised
methods were explored for this task.
Machine Translation
Statistical machine translation (SMT) systems emerged, which used
statistical models to translate text from one language to another.
Phrase-based and hierarchical models were common approaches in
SMT.
Information Retrieval
Statistical techniques were applied to information retrieval tasks,
where documents were ranked based on their relevance to user
queries.
While statistical approaches showed great promise, they still faced
challenges related to data sparsity, handling long-range dependencies
in language, and capturing complex semantic relationships between
words.
During the 2000s and 2010s, as we discussed in the history of AI,
there was a signiicant rise in the application of machine learning (ML)
techniques. This period witnessed tremendous advancements in ML
algorithms, computational power, and the availability of large text
corpora, which fueled the progress of NLP research and applications.
Several key developments contributed to the rise of machine
learning–based NLP during this time. Let us explore a few of them.
Statistical Approaches
Statistical approaches became dominant in NLP during this period.
Instead of hand-crafted rule-based systems, researchers started using
probabilistic models and ML algorithms to solve NLP tasks. Techniques
like HMMs, CRFs, and support vector machines (SVMs) gained
popularity.
Availability of Large Text Corpora
The rise of the Internet and digitalization led to the availability of vast
amounts of text data. Researchers could now train ML models on large
corpora, which greatly improved the performance of NLP systems.
Supervised Learning for NLP Tasks
Supervised learning became widely used for various NLP tasks. With
labeled data for tasks like part-of-speech tagging, named entity
recognition (NER), sentiment analysis, and machine translation,
researchers could train ML models effectively.
Named Entity Recognition
ML-based NER systems, which identify entities such as the names of
people, organizations, and locations in text, became more accurate and
widely used. This was crucial for information extraction and text
understanding tasks.
Sentiment Analysis
Sentiment analysis or opinion mining gained prominence, driven by the
increasing interest in understanding public opinions and sentiments
expressed in social media and product reviews.
Machine Translation
Statistical machine translation (SMT) systems, using techniques such as
phrase-based models, started to outperform rule-based approaches,
leading to signiicant improvements in translation quality.
Introduction of Word Embeddings
Word embeddings, like Word2Vec and GloVe, revolutionized NLP by
providing dense vector representations of words. These embeddings
captured semantic relationships between words, improving
performance in various NLP tasks.
Deep Learning and Neural Networks
The advent of deep learning and neural networks brought about a
paradigm shift in NLP. Models like recurrent neural networks (RNNs),
long short-term memory (LSTM), and convolutional neural networks
(CNNs) signiicantly improved performance in sequence-to-sequence
tasks, sentiment analysis, and machine translation.
Deployment in Real-World Applications
ML-based NLP systems found practical applications in various
industries, such as customer support chatbots, virtual assistants,
sentiment analysis tools, and machine translation services.
The combination of statistical methods, large datasets, and the
advent of deep learning paved the way for the widespread adoption of
ML-based NLP during the 2000s and 2010s.
Toward the end of the 2010s, pre-trained language models like
ELMo, Generative Pre-trained Transformer (GPT), and Bidirectional
Encoder Representations from Transformers (BERT) emerged. These
models were pre-trained on vast amounts of data and ine-tuned for
speciic NLP tasks, achieving state-of-the-art results in various
benchmarks. These developments enabled signiicant progress in
language understanding, text generation, and other NLP tasks, making
NLP an essential part of many modern applications and services."""]

# 2.Transformations
'''
   After the data is loaded, you then need to process and transform your data before putting it into a storage system.
   These transformations include chunking, extracting metadata, and embedding each chunk.
   This is necessary to make sure that the data can be retrieved, and used optimally by the LLM.
'''
from llama_index.embeddings.ollama import OllamaEmbedding
ollama_embedding = OllamaEmbedding(
    model_name="tinyllama:latest",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

pass_embedding = ollama_embedding.get_text_embedding_batch(
    text
)
print(pass_embedding)

# query_embedding = ollama_embedding.get_query_embedding(documents)
# print(query_embedding)