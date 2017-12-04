#!/bin/python3

import sys
import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
        '''Implement this module to extract
        and combine text files under train_path directory into 
        imdb_tr.csv. Each text file in train_path should be stored 
        as a row in imdb_tr.csv. And imdb_tr.csv should have two 
        columns, "text" and label'''
        pos_files = os.listdir(inpath+"/pos")
        neg_files = os.listdir(inpath+"/neg")
        outfile = open(outpath+name,"w")
        #outfile.write(',text,polarity\n')
        data = pd.DataFrame(columns=('text','polarity'))
        data.to_csv(outfile)
        count = 0
        for files in pos_files:
                with open(inpath+"/pos/"+files,'r') as f:
                        #outfile.write("%d,\'%s\',%d\n"%(count,f.read().replace("\"","\"\""),1))
                        #data.loc[count] = [f.read(),1]
                        row = pd.DataFrame([[f.read(),1]],columns=['text','polarity'],index=[count])
                        #data.append(row)
                        row.to_csv(outfile,header=False)
                        count += 1
                        #print(row)
        for files in neg_files:
                with open(inpath+"/neg/"+files,'r') as f:
                        #outfile.write("%d,\'%s\',%d\n"%(count,f.read().replace("\"","\"\""),0))
                        #data.loc[count] = [f.read(),0]
                        row = pd.DataFrame([[f.read(),0]],columns=['text','polarity'],index=[count])
                        #data.append(row)
                        count += 1
                        row.to_csv(outfile,header=False)
                        #print(row)
        #print(data)
        #data.to_csv(outfile)
        #outfile.close()

if __name__ == "__main__":
        imdb_data_preprocess(train_path)
        data =  pd.read_csv("./imdb_tr.csv",encoding="ISO-8859-1",index_col=0)
        train, val, train_target, val_target = train_test_split(data['text'].values,data['polarity'].values,test_size = 0.2,random_state=0)
        #print(train)
        #print(val)
        #print(train_target)
        #print(val_target)
        #train_target = train_target['polarity'].values.reshape((train_target.shape[0]))
        #val_target = val_target['polarity'].values.reshape((val.shape[0]))
        
        '''train a SGD classifier using unigram representation,
        predict sentiments on imdb_te.csv, and write output to
        unigram.output.txt'''
        with open('stopwords.en.txt') as f:
                stop_word = f.read().split("\n")
        test = pd.read_csv(test_path,encoding = 'ISO-8859-1', index_col = 0)
        #test = pd.read_csv(test_file,encoding="ISO-8859-1",index_col=0)
        #print(test_file)
        #test_target = test['polarity'].values.reshape((test.shape[0])) 
        #print(test)

        unigram = CountVectorizer(stop_words=stop_word)
        unigram_train = unigram.fit_transform(train)
        unigram_val = unigram.transform(val)
        unigram_test = unigram.transform(test['text'].values)

        params = [{'alpha':[0.0001,0.001,0.01,0.02,0.05,0.01],'penalty':['l1','l2']}]

        sgd = SGDClassifier(loss="hinge")
        clf = GridSearchCV(estimator=sgd,param_grid=params,n_jobs=-1,cv=5)
        clf.fit(unigram_train,train_target)
        train_pre = clf.predict(unigram_train)
        print(accuracy_score(train_target,train_pre))
        val_pre = clf.predict(unigram_val)
        print(accuracy_score(val_target,val_pre))
        test_pre = clf.predict(unigram_test)
        np.savetxt('unigram.output.txt',test_pre,fmt="%d",delimiter="/n")
        print(test_pre)

        '''train a SGD classifier using unigram representation
        with tf-idf, predict sentiments on imdb_te.csv, and write
        output to unigramtfidf.output.txt'''

        unigramT = TfidfVectorizer(stop_words=stop_word)
        unigramT_train = unigramT.fit_transform(train)
        unigramT_val = unigramT.transform(val)
        unigramT_test = unigramT.transform(test['text'].values)

        clf.fit(unigramT_train,train_target)
        train_pre = clf.predict(unigramT_train)
        print(accuracy_score(train_target,train_pre))
        val_pre = clf.predict(unigramT_val)
        print(accuracy_score(val_target,val_pre))
        test_pre = clf.predict(unigramT_test)
        np.savetxt('unigramtfidf.output.txt',test_pre,fmt="%d",delimiter="/n")
        print(test_pre)

        '''train a SGD classifier using bigram representation,
        predict sentiments on imdb_te.csv, and write output t1
        bigram.output.txt'''


        bigram = CountVectorizer(stop_words=stop_word,ngram_range=(1,2))
        bigram_train = bigram.fit_transform(train)
        bigram_val = bigram.transform(val)
        bigram_test = bigram.transform(test['text'].values)

        params = [{'alpha':[0.0001,0.001,0.01,0.02,0.05,0.01],'penalty':['l1','l2']}]

        clf.fit(bigram_train,train_target)
        train_pre = clf.predict(bigram_train)
        print(accuracy_score(train_target,train_pre))
        val_pre = clf.predict(bigram_val)
        print(accuracy_score(val_target,val_pre))
        test_pre = clf.predict(bigram_test)
        np.savetxt('bigram.output.txt',test_pre,fmt="%d",delimiter="/n")
        print(test_pre)

        '''train a SGD classifier using bigram representation
        with tf-idf, predict sentiments on imdb_te.csv, and write
        output to bigramtfidf.output.txt'''

        bigramT = TfidfVectorizer(stop_words=stop_word,ngram_range=(1,2))
        bigramT_train = bigramT.fit_transform(train)
        bigramT_val = bigramT.transform(val)
        bigramT_test = bigramT.transform(test['text'].values)

        clf.fit(bigramT_train,train_target)
        train_pre = clf.predict(bigramT_train)
        print(accuracy_score(train_target,train_pre))
        val_pre = clf.predict(bigramT_val)
        print(accuracy_score(val_target,val_pre))
        test_pre = clf.predict(bigramT_test)
        np.savetxt('bigramtfidf.output.txt',test_pre,fmt="%d",delimiter="/n")
        print(test_pre)

