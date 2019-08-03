import os
import pickle
import re
import math

path_to_dir1 = 'D:/ham_train'
path_to_dir2 = 'D:/spam_train'
D0 = 0  # ham
D1 = 0  # spam
ham_words = 0
spam_words = 0
ham_dict = {}
spam_dict = {}

files = os.listdir(path_to_dir1)
for j in files:
    D0 += 1

    path_to_file = path_to_dir1 + '/' + j
    f = open(path_to_file, 'r', encoding='ansi')
    text = f.read()
    text = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', "", text)
    words = text.split()
    f.close()
    for i in words:
        ham_words += 1
        ham_dict[i] = ham_dict.get(i, 0) + 1
path_ham = 'ham_dict.bin'
f = open(path_ham, 'wb')
pickle.dump(ham_dict, f)
f.close()
path_ham3 = 'ham_words.bin'
f = open(path_ham3, 'wb')
pickle.dump(ham_words, f)
f.close()
path_ham4 = 'd0.bin'
f = open(path_ham4, 'wb')
pickle.dump(D0, f)
f.close()


files = os.listdir(path_to_dir2)
for j in files:
    D1 += 1
    path_to_file = path_to_dir2 + '/' + j
    f = open(path_to_file, 'r', encoding='ansi')
    text = f.read()
    text = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', "", text)
    words = text.split()
    f.close()
    for i in words:
        spam_words += 1
        spam_dict[i] = spam_dict.get(i, 0) + 1

path_spam = 'spam_dict.bin'
f = open(path_spam, 'wb')
pickle.dump(spam_dict, f)
f.close()
path_spam3 = 'spam_words.bin'
f = open(path_spam3, 'wb')
pickle.dump(spam_words, f)
f.close()
path_spam4 = 'd1.bin'
f = open(path_spam4, 'wb')
pickle.dump(D1, f)
f.close()

dict_unique = len(ham_dict)
for i in spam_dict:
    if i in list(ham_dict.keys()):
        dict_unique += 0
    else:
        dict_unique += 1
        
path = 'dict_unique.bin'
f = open(path, 'wb')
pickle.dump(dict_unique, f)
f.close()


path_ham = 'ham_dict.bin'
path_ham3 = 'ham_words.bin'
path_ham4 = 'd0.bin'
g = open(path_ham, 'rb')
dict_ham = pickle.load(g)
g.close()
g = open(path_ham3, 'rb')
ham_words = pickle.load(g)
g.close()
g = open(path_ham4, 'rb')
D0 = pickle.load(g)
g.close()

path_spam = 'spam_dict.bin'
path_spam3 = 'spam_words.bin'
path_spam4 = 'd1.bin'
g = open(path_spam, 'rb')
dict_spam = pickle.load(g)
g.close()
g = open(path_spam3, 'rb')
spam_words = pickle.load(g)
g.close()
g = open(path_spam4, 'rb')
D1 = pickle.load(g)
g.close()

path = 'dict_unique.bin'
g = open(path, 'rb')
dict_unique = pickle.load(g)
g.close()

path_to_ham = 'D:/ham_test'
path_to_spam = 'D:/spam_test'

TP = 0
FN = 0
TN = 0
FP = 0

files = os.listdir(path_to_ham)
for i in files:
    help_c0 = 0
    help_c1 = 0
    path_to_file = path_to_ham + '/' + i
    f = open(path_to_file, 'r', encoding='ansi')
    text = f.read()
    text = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', "", text)
    words = text.split()
    f.close()
    for j in words:
        F0 = dict_ham.get(j, 0)
        F1 = dict_spam.get(j, 0)

        help_c0 += math.log((F0 + 1) / (dict_unique + ham_words))
        help_c1 += math.log((F1 + 1) / (dict_unique + spam_words))
    c0 = math.log(D0 / (D0 + D1)) + help_c0
    c1 = math.log(D1 / (D0 + D1)) + help_c1
    t = 0.8
    k = -c0
    p1 = math.exp(c1 + k) / (math.exp(c0 + k) + math.exp(c1 + k))
    if t < p1:
        FN += 1
    else:
        TP += 1

files = os.listdir(path_to_spam)
for i in files:
    help_c0 = 0
    help_c1 = 0
    path_to_file = path_to_spam + '/' + i
    f = open(path_to_file, 'r', encoding='ansi')
    text = f.read()
    text = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', "", text)
    words = text.split()
    f.close()
    for j in words:
        F0 = dict_ham.get(j, 0)
        F1 = dict_spam.get(j, 0)
        help_c0 += math.log((F0 + 1) / (dict_unique + ham_words))
        help_c1 += math.log((F1 + 1) / (dict_unique + spam_words))
    c0 = math.log(D0 / (D0 + D1)) + help_c0
    c1 = math.log(D1 / (D0 + D1)) + help_c1
    t = 0.8
    k = -c1
    p1 = math.exp(c1 + k)/(math.exp(c0 + k) + math.exp(c1 + k))
    if t < p1:
        TN += 1
    else:
        FP += 1

N = TP + FP + TN + FN
Accuracy = (TP + TN) / N

print(TP)
print(FN)
print(TN)
print(FP)
print(Accuracy)