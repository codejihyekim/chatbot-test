import itertools

import matplotlib
from konlpy.tag import Okt
from matplotlib import font_manager, rc
from nltk.tokenize import word_tokenize
import nltk
import re
import pandas as pd
import numpy as np
from nltk import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from icecream import ic
from collections import Counter, defaultdict
from context.domains import File, Reader

class Solution(Reader):
    def __init__(self):
        self.file = File()
        self.file.context = './data/'
        self.okt = Okt()

    def h_preprocessing(self):
        df = pd.read_excel('./data/감성대화말뭉치(원시데이터)_Validation.xlsx', engine='openpyxl')
        df_happy = df.drop(
            ['번호', 'value', '연령', '성별', '상황키워드', '신체질환', '감정_소분류', '시스템응답1', '사람문장2', '시스템응답2', '사람문장3', '시스템응답3'],
            axis=1)
        df_happy = df_happy[df_happy['감정_대분류'] == '기쁨']
        df_happy = df_happy.drop(['감정_대분류'], axis=1)
        df_happy.rename(columns={'사람문장1': 'doc'}, inplace=True)
        df_happy = np.array(df_happy)
        df_happy = list(itertools.chain(*df_happy))
        happy_texts = ''
        for i in df_happy:
            happy_texts += i + ""
        happy_texts = happy_texts.replace('\n', ' ')
        tokenizer = re.compile(r'[^ㄱ-힣]+')
        #print(tokenizer.sub(' ', happy_texts))
        return tokenizer.sub(' ', happy_texts)

    def s_preprocessing(self):
        df = pd.read_excel('./data/감성대화말뭉치(원시데이터)_Validation.xlsx', engine='openpyxl')
        df_sad = df.drop(
            ['번호', 'value', '연령', '성별', '상황키워드', '신체질환', '감정_소분류', '시스템응답1', '사람문장2', '시스템응답2', '사람문장3',
             '시스템응답3'],
            axis=1)
        df_sad = df_sad[df_sad['감정_대분류'] == '슬픔']
        df_sad = df_sad.drop(['감정_대분류'], axis=1)
        df_sad.rename(columns={'사람문장1': 'doc'}, inplace=True)
        df_sad = np.array(df_sad)
        df_sad = list(itertools.chain(*df_sad))
        sad_texts = ''
        for i in df_sad:
            sad_texts += i + ""
        sad_texts = sad_texts.replace('\n', ' ')
        tokenizer = re.compile(r'[^ㄱ-힣]+')
        return tokenizer.sub(' ', sad_texts)

    def a_preprocessing(self):
        df = pd.read_excel('./data/감성대화말뭉치(원시데이터)_Validation.xlsx', engine='openpyxl')
        df_angry = df.drop(
            ['번호', 'value', '연령', '성별', '상황키워드', '신체질환', '감정_소분류', '시스템응답1', '사람문장2', '시스템응답2', '사람문장3',
             '시스템응답3'],
            axis=1)
        df_angry = df_angry[df_angry['감정_대분류'] == '분노']
        df_angry = df_angry.drop(['감정_대분류'], axis=1)
        df_angry.rename(columns={'사람문장1': 'doc'}, inplace=True)
        df_angry = np.array(df_angry)
        df_angry = list(itertools.chain(*df_angry))
        angry_texts = ''
        for i in df_angry:
            angry_texts += i + ""
        angry_texts = angry_texts.replace('\n', ' ')
        tokenizer = re.compile(r'[^ㄱ-힣]+')
        return tokenizer.sub(' ', angry_texts)

    def u_preprocessing(self):
        df = pd.read_excel('./data/감성대화말뭉치(원시데이터)_Validation.xlsx', engine='openpyxl')
        df_angry = df.drop(
            ['번호', 'value', '연령', '성별', '상황키워드', '신체질환', '감정_소분류', '시스템응답1', '사람문장2', '시스템응답2', '사람문장3',
             '시스템응답3'],
            axis=1)
        df_angry = df_angry[df_angry['감정_대분류'] == '불안']
        df_angry = df_angry.drop(['감정_대분류'], axis=1)
        # print(df_angry)
        df_angry.rename(columns={'사람문장1': 'doc'}, inplace=True)
        df_angry = np.array(df_angry)
        # print(df_angry)
        df_angry = list(itertools.chain(*df_angry))
        # print(df_angry)
        angry_texts = ''
        for i in df_angry:
            angry_texts += i + ""
        angry_texts = angry_texts.replace('\n', ' ')
        # print(len(angry_texts))
        tokenizer = re.compile(r'[^ㄱ-힣]+')
        #print(tokenizer.sub(' ', angry_texts))
        return tokenizer.sub(' ', angry_texts)

    def happy_stopword(self):
        file = self.file
        file.fname = 'happy_stopword.txt'
        path = self.new_file(file)
        with open(path, 'r', encoding='utf-8') as f:
            texts = f.read()
        return texts.strip()

    def sad_stopword(self):
        file = self.file
        file.fname = 'sad_stopword.txt'
        path = self.new_file(file)
        with open(path, 'r', encoding='utf-8') as f:
            texts = f.read()
        return texts.strip()

    def angry_stopword(self):
        file = self.file
        file.fname = 'angry_stropword.txt'
        path = self.new_file(file)
        with open(path, 'r', encoding='utf-8') as f:
            texts = f.read()
        return texts.strip()

    def unrest_stopword(self):
        file = self.file
        file.fname = 'unrest_stopword.txt'
        path = self.new_file(file)
        with open(path, 'r', encoding='utf-8') as f:
            texts = f.read()
        return texts.strip()

    def h_noun_embedding(self):
        stopword = self.happy_stopword()
        noun_tokens = []
        tokens = word_tokenize(self.h_preprocessing())
        for i in tokens:
            pos = self.okt.pos(i)
            _ = [j[0] for j in pos if j[1] in ['Verb', 'Adverb', 'Adjective', 'Noun']]
            #print(_)
            if len(''.join(_)) > 1:
                noun_tokens.append(' '.join(_))
        morphemes = [text for text in noun_tokens if text not in stopword]
        return morphemes

    def s_noun_embedding(self):
        stopword = self.sad_stopword()
        noun_tokens = []
        tokens = word_tokenize(self.s_preprocessing())
        for i in tokens:
            pos = self.okt.pos(i)
            _ = [j[0] for j in pos if j[1] in ['Verb', 'Adverb', 'Adjective', 'Noun']]
            if len(''.join(_)) > 1:
                noun_tokens.append(' '.join(_))
        morphemes = [text for text in noun_tokens if text not in stopword]
        return morphemes

    def a_noun_embedding(self):
        stopword = self.angry_stopword()
        noun_tokens = []
        tokens = word_tokenize(self.a_preprocessing())
        for i in tokens:
            pos = self.okt.pos(i)
            _ = [j[0] for j in pos if j[1] in ['Verb', 'Adverb', 'Adjective', 'Noun']]
            if len(''.join(_)) > 1:
                noun_tokens.append(' '.join(_))
        morphemes = [text for text in noun_tokens if text not in stopword]
        return morphemes

    def u_noun_embedding(self):
        stopword = self.unrest_stopword()
        noun_tokens = []
        tokens = word_tokenize(self.u_preprocessing())
        for i in tokens:
            pos = self.okt.pos(i)
            _ = [j[0] for j in pos if j[1] in ['Verb', 'Adverb', 'Adjective', 'Noun']]
            if len(''.join(_)) > 1:
                noun_tokens.append(' '.join(_))
        morphemes = [text for text in noun_tokens if text not in stopword]
        return morphemes

    def h_draw_wordcloud(self):
        _ = self.h_noun_embedding()
        freqtxt = pd.Series(dict(FreqDist(_))).sort_values(ascending=False)
        ic(freqtxt)
        wcloud = WordCloud('./data/D2Coding.ttf', relative_scaling=0.2,
                           background_color='white').generate(" ".join(_))
        plt.figure(figsize=(20, 20))
        plt.imshow(wcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def s_draw_wordcloud(self):
        _ = self.s_noun_embedding()
        freqtxt = pd.Series(dict(FreqDist(_))).sort_values(ascending=False)
        wcloud = WordCloud('./data/D2Coding.ttf', relative_scaling=0.2,
                           background_color='white').generate(" ".join(_))
        plt.figure(figsize=(12, 12))
        plt.imshow(wcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def a_draw_wordcloud(self):
        _ = self.a_noun_embedding()
        freqtxt = pd.Series(dict(FreqDist(_))).sort_values(ascending=False)
        wcloud = WordCloud('./data/D2Coding.ttf', relative_scaling=0.2,
                           background_color='white').generate(" ".join(_))
        plt.figure(figsize=(12, 12))
        plt.imshow(wcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def u_draw_wordcloud(self):
        _ = self.u_noun_embedding()
        freqtxt = pd.Series(dict(FreqDist(_))).sort_values(ascending=False)
        wcloud = WordCloud('./data/D2Coding.ttf', relative_scaling=0.2,
                           background_color='white').generate(" ".join(_))
        plt.figure(figsize=(12, 12))
        plt.imshow(wcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def visualization(self):
        self.ko_font()
        #기쁨 단어
        h_texts = self.h_noun_embedding()
        happy_count = Counter(h_texts)
        max = 20
        happy_top_20 = {}
        for word, counts in happy_count.most_common(max):
            happy_top_20[word] = counts
            print(f'{word} : {counts}')
        plt.figure(figsize=(10,5))
        plt.title('기쁨 단어 상위 (%d개)' % max, fontsize=17)
        plt.ylabel('단어의 빈도수')
        plt.xticks(rotation=70)
        for key, value in happy_top_20.items():
            plt.bar(key, value, color='lightgrey')
        plt.show()

        print('*'*50)
        #슬픔 단어
        s_texts = self.s_noun_embedding()
        sad_count = Counter(s_texts)
        max = 20
        sad_top_20 = {}
        for word, counts in sad_count.most_common(max):
            sad_top_20[word] = counts
            print(f'{word} : {counts}')
        plt.figure(figsize=(10, 5))
        plt.title('슬픔 단어 상위 (%d개)' % max, fontsize=17)
        plt.ylabel('단어의 빈도수')
        plt.xticks(rotation=70)
        for key, value in sad_top_20.items():
            plt.bar(key, value, color='lightgrey')
        plt.show()

        print('*' * 50)
        # 분노 단어
        a_texts = self.a_noun_embedding()
        angry_count = Counter(a_texts)
        max = 20
        angry_top_20 = {}
        for word, counts in angry_count.most_common(max):
            angry_top_20[word] = counts
            print(f'{word} : {counts}')
        plt.figure(figsize=(10, 5))
        plt.title('분노 단어 상위 (%d개)' % max, fontsize=17)
        plt.ylabel('단어의 빈도수')
        plt.xticks(rotation=70)
        for key, value in angry_top_20.items():
            plt.bar(key, value, color='lightgrey')
        plt.show()

        print('*' * 50)
        # 불안 단어
        u_texts = self.u_noun_embedding()
        unrest_count = Counter(u_texts)
        max = 20
        unrest_top_20 = {}
        for word, counts in unrest_count.most_common(max):
            unrest_top_20[word] = counts
            print(f'{word} : {counts}')
        plt.figure(figsize=(10, 5))
        plt.title('불안 단어 상위 (%d개)' % max, fontsize=17)
        plt.ylabel('단어의 빈도수')
        plt.xticks(rotation=70)
        for key, value in unrest_top_20.items():
            plt.bar(key, value, color='lightgrey')
        plt.show()

    def ko_font(self):
        font_path = "C:/Windows/Fonts/malgunsl.ttf"
        font = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font)
        # 그래프 마이너스 기호 표시 설정
        matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    s = Solution()
    #s.u_noun_embedding()
    s.visualization()
    #s.h_preprocessing()
    #s.h_noun_embedding()
    #s.h_draw_wordcloud()
    #s.s_draw_wordcloud()
    #s.a_draw_wordcloud()
    #s.u_draw_wordcloud()