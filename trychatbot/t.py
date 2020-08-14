import spacy
import nltk
from nltk.corpus import treebank

# texts = [
#     "d'água",
#     "Sim, nós conhecemos na faculdade ele tinha o sonho de ter um próprio jogo dele",
#     "Bom dia",
#     "Fala aê",
#     "Qual é a boa?",
#     "Tudo tranquilo por aqui, e aí?",
#     "Vire-se"
# ]

# nlp = spacy.load("pt_core_news_sm")

# from spacy.lang.pt import Portuguese
# from spacy.tokenizer import Tokenizer

# parser = Portuguese()

# tokenizer = Tokenizer(parser.vocab)



# for t in texts:
#     tks = nlp(t)
#     _tokens = nltk.word_tokenize(t)
#     print([tk for tk in _tokens])
#     # tokens = parser(t)
#     # for tk in tks:
#         # print(tk.sentiment)
#     print("***")

nlp = spacy.load("pt_core_news_sm")

sentence = "João tem um cão"

doc = nlp(sentence)

for token in doc:
    print("{} ### {} ### {} ### {} ### {} ### {} ### {} ### {}".format(token.text, token.lemma_, token.pos_, token.tag_,
token.dep_, token.shape_, token.is_alpha, token.is_stop))

exit()

taggs = nltk.pos_tag(tokens)

print(taggs)

entities = nltk.chunk.ne_chunk(taggs)

print(entities)

st = nltk.PorterStemmer()
l = [st.stem(t) for t in nltk.word_tokenize("principalmente")]

treebank.parse

# print(l)



