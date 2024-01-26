import re
import subprocess
import time
import streamlit as st
from  PIL import Image
import numpy as np
import pandas as pd
import io
import os
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def main():
    client = MongoClient("mongodb://localhost:27017")
    db = client["CONNECT-X"]
    collection = db["test"]
    documents = collection.find()
    
    df = pd.DataFrame(columns=['Query', 'Response_A', 'Response_B', 'JACCARD', 'A vs B', '평가자'])

    for documents in documents:
        query = documents.get('query')
        response_a = documents.get('Response = A\n(12', {}).get('19', {}).get('12시 - Prod)')
        response_b = documents.get('Response = B\n(12', {}).get('07', {}).get('14시 - Stage, 사내검색 컨텐츠 Vector DB 재색인)')
        jaccard = documents.get('JACCARD\n유사도')
        winLoss = documents.get('A vs B\n\n우위 응답')
        tester =  documents.get('평가자')

        if query is not None and response_a is not None and response_b is not None:
                df.loc[len(df)] = [query, response_a, response_b, jaccard, winLoss, tester]

#    st.write("View Entire Data")
#    st.write(df)


    selected_row = st.number_input('Insert a number', min_value=0, max_value=df.shape[0]-1)
    selected_data = df.loc[selected_row]

    st.write(df.loc[selected_row]['Query'])

    similarity_score = jaccard_similarity(df.loc[selected_row]['Response_A'], df.loc[selected_row]['Response_B'])
    st.write(''':orange[Jaccard similarity score :]''', round(similarity_score, 5))

    similarity = sentence_similarity(df.loc[selected_row]['Response_A'], df.loc[selected_row]['Response_B'])
    st.write(''':orange[Cosine similarity score :]''', round(similarity, 5))
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('''***:blue[Response A - Prod]***''')
        st.caption(df.loc[selected_row]['Response_A'])

    with col2:
        st.markdown('''***:green[Response B - Stage]***''')
        st.caption(df.loc[selected_row]['Response_B'])

    st.divider()
    st.write("View Entire Data")
    st.write(df)









def jaccard_similarity(sentence1, sentence2):
    # 문장을 단어로 분할하여 집합으로 변환
    set1 = set(sentence1.split())
    set2 = set(sentence2.split())

    # 교집합과 합집합의 크기 계산
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))

    # Jaccard 유사도 계산
    if union_size == 0:
        return 0.0
    else:
        return intersection_size / union_size




def sentence_similarity(sentence1, sentence2):
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([sentence1, sentence2])
    
    # 코사인 유사도 계산
    similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]
    
    return similarity_score







if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
