import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from tqdm import tqdm
import re
# seperate title [ 1. 리뷰 파일 다운로드 ] ==========================================#
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

# seperate title [ 2. Pandas 데이터 확인 ] ==========================================#
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')
print(train_data.info())
print(test_data.info())
#>>> 출력결과
#>>> train_data
#>>> <class 'pandas.core.frame.DataFrame'>
#>>>  #   Column    Non-Null Count   Dtype 
#>>> ---  ------    --------------   ----- 
#>>>  0   id        150000 non-null  int64 
#>>>  1   document  149995 non-null  object
#>>>  2   label     150000 non-null  int64 
#>>> dtypes: int64(2), object(1)
#>>> test_data
#>>> <class 'pandas.core.frame.DataFrame'>
#>>>  #   Column    Non-Null Count  Dtype 
#>>> ---  ------    --------------  ----- 
#>>>  0   id        50000 non-null  int64 
#>>>  1   document  49997 non-null  object
#>>>  2   label     50000 non-null  int64 
#>>> dtypes: int64(2), object(1)
# < 개발자 분석내용 > - document 필드의 수량이 label 필드의 수량과 다르므로 결측 데이터가 존재한다

# seperate title [ 3. 결측 데이터 수량 확인 및 제거 ] ==========================================#
# document 필드 결측치 수량
print("훈련데이터 결측치 수량",train_data["document"].isna().sum())
print("테스트데이터 결측치 수량",test_data["document"].isna().sum())
# 결측값 제거 후 결측치 수량 확인
train_data = train_data.dropna(subset = ["document"]) # (axis=0, subset=["document"])
test_data = test_data.dropna(subset = ["document"]) # (axis=0, subset=["document"])
print("훈련데이터 결측치 수량",train_data["document"].isna().sum())
print("훈련데이터 결측치 수량",test_data["document"].isna().sum())
#>>> 출력결과
#>>> 훈련데이터 결측치 수량 5
#>>> 테스트데이터 결측치 수량 3
#>>> 훈련데이터 결측치 수량 0
#>>> 훈련데이터 결측치 수량 0
# < 개발자 분석내용 > - 훈련데이터의 결측데이터 5개, 테스트데이터의 결측데이터 3개가 확인되어 pandas의 dropna 명령으로 제거

# seperate title [ 4. 중복 데이터 확인 및 제거 ] ==========================================#
# document 중복 검사
print(train_data["document"].count()) # 총 데이터 수량
print(train_data["document"].nunique()) # 유일한 데이터 수량
#>>> 출력결과
#>>> 149995
#>>> 146182
# < 개발자 분석내용 > - count와 nunique의 수량이 차이가 있기 때문에 중복 데이터가 존재함 ( 3813개 )
print(test_data["document"].count())
print(test_data["document"].nunique())
#>>> 출력결과
#>>> 49997
#>>> 49157
#>>> < 개발자 분석내용 > - count와 nunique의 수량 차이가 있기 때문에 중복 데이터 존재함
#>>> 훈련대상 데이터가 아니지만 중복 내용을 제거함.

# 중복 데이터 제거
# ref참조 DataFrame.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
train_data = train_data.drop_duplicates(subset=["document"])
test_data = test_data.drop_duplicates(subset=["document"])
print("중복된 훈련데이터의 수 =", train_data["document"].count() - train_data["document"].nunique())
print("중복된 테스트데이터의 수 =", test_data["document"].count() - test_data["document"].nunique())
#>>> 중복된 훈련데이터의 수 = 0
#>>> 중복된 테스트데이터의 수 = 0
# < 개발자 분석내용 > - 중복된 데이터가 제거되어 0으로 표기됨

# seperate title [ 5. 한글을 제외한 문자 제거 및 형태소 분류 ] ==========================================#
train_data["document"] = train_data["document"].replace('[^\\sㄱ-ㅎㅏ-ㅣ가-힣]',"",regex=True)
test_data["document"] = test_data["document"].replace('[^\\sㄱ-ㅎㅏ-ㅣ가-힣]',"",regex=True)
print(train_data[:5])
#>>> 출력결과
#>>>          id                                           document  label
#>>> 0   9976970                                  아 더빙 진짜 짜증나네요 목소리      0
#>>> 1   3819312                         흠포스터보고 초딩영화줄오버연기조차 가볍지 않구나      1
#>>> 2  10265843                                  너무재밓었다그래서보는것을추천한다      0
#>>> 3   9045019                          교도소 이야기구먼 솔직히 재미는 없다평점 조정      0
#>>> 4   6483659  사이몬페그의 익살스런 연기가 돋보였던 영화스파이더맨에서 늙어보이기만 했던 커스틴 던...      1
# < 개발자 분석내용 > - 정규표현식을 이용해 한글과 공백을 제외한 모든 단어를 제거함 ( , . / ? 및 영문 등 )
