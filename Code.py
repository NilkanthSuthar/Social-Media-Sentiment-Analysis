
from bs4 import BeautifulSoup as bs
import requests
import glob
from nltk.tokenize import word_tokenize, sent_tokenize
import syllables
import pandas as pd

df = pd.read_excel('Input.xlsx')
urls = df.iloc[:,1].tolist()
url_id = df.iloc[:,0].tolist()
for url, uid in zip(urls, url_id):
    response = requests.get(url)
    soup = bs(response.content, 'html.parser')
    
    title_container = soup.find('h1')
    if title_container is not None:
        title = title_container.get_text().strip()
        article_container = soup.find('article')
        if article_container is not None:
            article = '\n\n'.join([p.get_text().strip() for p in article_container.select('p')])
            with open(f'Output//{uid}.txt', 'w', encoding='utf-8') as f:
                f.write(f'{title}\n\n{article}')

pos_words = open('MasterDictionary\\positive-words.txt', 'r').read().splitlines()
neg_words = open('MasterDictionary\\negative-words.txt', 'r').read().splitlines()
stop_words=[]
file_list = glob.glob('StopWords\*.txt')
for file in file_list:
    with open(file, 'r') as f:
        txt = f.read().splitlines()
        stop_words +=txt

def sentiment_analysis(text):
    
    words = word_tokenize(text)
    sentences = sent_tokenize(article_text)
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    words = [word for word in words if not word in stop_words]
    pos_count = len([word for word in words if word in pos_words])
    neg_count = len([word for word in words if word in neg_words])
    polarity_score = (pos_count - neg_count) / ((pos_count + neg_count) + 0.000001)
    subjectivity_score = (pos_count + neg_count) / ((len(words)) + 0.000001)
    syllables_per_word = sum([syllables.estimate(word) for word in words]) / len(words)
    avg_word_length = sum([len(word) for word in words]) / len(words)
    word_count = len(words)
    avg_sentence_length = word_count / text.count('.') 
    complex_word_count = len([word for word in words if len(word) > 2 and word.isalpha() and word not in pos_words and word not in neg_words])
    percent_complex_words = complex_word_count / word_count
    fog_index = 0.4 * (avg_sentence_length + percent_complex_words)
    avg_word_length = sum(len(word) for word in words) / word_count
    personal_pronouns = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    personal_pronoun_count = len([word for word in words if word.lower() in personal_pronouns])
    
    return [pos_count, neg_count, polarity_score, subjectivity_score, 
            avg_sentence_length, percent_complex_words, fog_index, len(words)/len(sentences),
            complex_word_count, word_count, syllables_per_word, personal_pronoun_count, avg_word_length]


results = []
df = pd.read_excel('Input.xlsx')
for i, row in df.iterrows():
    with open(f'Output\\{row["URL_ID"]}.txt', 'r', encoding='utf-8') as f:
        article_text = f.read()
    variables = sentiment_analysis(article_text)
    results.append([row['URL_ID']] +[row['URL']]+variables)

output=['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']
output_df = pd.DataFrame(columns=output)

for row in results:
    output_df = output_df.append(pd.Series(row, index=output), ignore_index=True)

output_df.to_excel('Output.xlsx', index=False)


