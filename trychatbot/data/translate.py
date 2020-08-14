# from googletrans import Translator
from six.moves import cPickle
# import os

# data = cPickle.load(open("data/alexa/all_convos.pkl","rb"))
# data = data[:5000]
# print(len(data))
# # inputs = [item[0] for item in data]
# # targets = [item[1] for item in data]

# t = Translator()
# all_convos=[]
# error_line=None
# for item in data:
#     input = t.translate(item[0], src="en", dest="pt").text
#     if item[1]!="":
#         target = t.translate(item[1], src="en", dest="pt").text
#         all_convos.append((input,target))

# cPickle.dump(all_convos,open("data/alexa/all_convos_translated.pkl","wb"))

data1 = cPickle.load(open("data/alexa/all_convos_translated.pkl","rb"))
data2 = cPickle.load(open("data/portugues/all_convos.pkl","rb"))

data2 = data2 + data1

print(data2[1:200])

cPickle.dump(data2,open("data/portugues/all_convos.pkl","wb"))