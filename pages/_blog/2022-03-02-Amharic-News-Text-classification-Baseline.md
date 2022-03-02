```python
%matplotlib inline

import matplotlib
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

np.random.seed(42) 
```

# Exploratory Data Analysis


```python
data = pd.read_csv('data/Amharic News Dataset.csv')

data = shuffle(data)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>headline</th>
      <th>category</th>
      <th>date</th>
      <th>views</th>
      <th>article</th>
      <th>link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15168</th>
      <td>ህወሃት ከፌዴራሊስት ሀይሎች ጋር እየሰራሁ ነው በማለት ህብረተሰቡን እያወ...</td>
      <td>ሀገር አቀፍ ዜና</td>
      <td>26-Sep-20</td>
      <td>2,204</td>
      <td>አዲስ አበባ፣ መስከረም 16፣ 2013 (ኤፍ.ቢ.ሲ) ህወሃት ከፌዴራሊስት ...</td>
      <td>https://www.fanabc.com/%e1%88%85%e1%8b%88%e1%8...</td>
    </tr>
    <tr>
      <th>13721</th>
      <td>አቶ ደመቀ መኮንን ከተመድ የአፍሪካ ቀንድ ልዩ መልእክተኛ ጋር ተወያዩ</td>
      <td>ሀገር አቀፍ ዜና</td>
      <td>1-Dec-20</td>
      <td>336</td>
      <td>አዲስ አበባ፣ ህዳር 22፣ 2013 (ኤፍ.ቢ.ሲ) የኢፌዴሪ ምክትል ጠቅላይ...</td>
      <td>https://www.fanabc.com/%e1%8a%a0%e1%89%b6-%e1%...</td>
    </tr>
    <tr>
      <th>21287</th>
      <td>በአዲስ ዓመት ዋዜማ አራት ሰዎች በማይታወቁ ተሽከርካሪዎች ተገጭተው ሞቱ</td>
      <td>ፖለቲካ</td>
      <td>15-Sep-19</td>
      <td>Unknown</td>
      <td>የ2011 ዓ.ም. ያለ ምንም የወንጀል ድርጊት በሰላም መጠናቀቁን ፖሊስ ቢ...</td>
      <td>https://www.ethiopianreporter.com/article/16750</td>
    </tr>
    <tr>
      <th>16264</th>
      <td>“የህዳሴ ግድቡ በወቅቱ በነበሩ የቦርድ ሰብሳቢ አድራጊ ፈጣሪነት ለከፍተኛ...</td>
      <td>ሀገር አቀፍ ዜና</td>
      <td>18-Jun-20</td>
      <td>1,751</td>
      <td>አዲስ አበባ፣ ሰኔ 11፣ 2012 (ኤፍ. ቢ.ሲ) የህዳሴ ግድብ ግንባታ በ...</td>
      <td>https://www.fanabc.com/%e1%8b%a8%e1%88%85%e1%8...</td>
    </tr>
    <tr>
      <th>24513</th>
      <td>በኦሮሚያ ክልል ንብረቶቻቸው የወደሙባቸው ኢንቨስተሮች የካሳ ጥያቄ እያቀረ...</td>
      <td>ፖለቲካ</td>
      <td>23-Oct-16</td>
      <td>Unknown</td>
      <td>በመስከረም ወር መጨረሻ በኦሮሚያ ክልል በተነሳው ሁከት ንብረቶቻቸው የወደ...</td>
      <td>https://www.ethiopianreporter.com/content/%E1%...</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 50708 entries, 15168 to 15795
    Data columns (total 6 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   headline  50694 non-null  object
     1   category  50706 non-null  object
     2   date      50707 non-null  object
     3   views     50707 non-null  object
     4   article   50707 non-null  object
     5   link      50706 non-null  object
    dtypes: object(6)
    memory usage: 2.7+ MB



```python
data = data.dropna(subset=['article'])
```


```python
data['link'].value_counts()
```




    https://amharic.voanews.com//a/ethiopia-tigrai-mekele/5679989.html                                                                                                                                                                                                                                                                                                        3
    https://amharic.voanews.com//a/rev-jesse-jackson-letter-to-hon-karen-bass-about-nile-river-5-21-2020o/5430577.html                                                                                                                                                                                                                                                        3
    https://amharic.voanews.com//a/white-house-on-river-nile-and-ethiopias-dam-10-03-19/5110165.html                                                                                                                                                                                                                                                                          3
    https://amharic.voanews.com//a/sudan-laws/5499793.html                                                                                                                                                                                                                                                                                                                    3
    https://amharic.voanews.com//a/covid-ethiopia/5470709.html                                                                                                                                                                                                                                                                                                                3
                                                                                                                                                                                                                                                                                                                                                                             ..
    https://am.al-ain.com/article/an-interim-experts-summit-is-to-be-formed-to-resolve-differences-among-olf-members                                                                                                                                                                                                                                                          1
    https://soccerethiopia.net/football/7715                                                                                                                                                                                                                                                                                                                                  1
    https://waltainfo.com/am/29558/                                                                                                                                                                                                                                                                                                                                           1
    https://www.addisadmassnews.com/index.php?option=com_k2&view=item&id=20922:%E1%89%A02050-%E1%8A%A8%E1%8A%A0%E1%88%88%E1%88%9B%E1%89%BD%E1%8A%95-%E1%88%85%E1%8B%9D%E1%89%A5-%E1%88%A9%E1%89%A5-%E1%8B%AB%E1%88%85%E1%88%89-%E1%8A%A0%E1%8D%8D%E1%88%AA%E1%8A%AB%E1%8B%8D%E1%8B%AB%E1%8A%95-%E1%8B%AD%E1%88%86%E1%8A%93%E1%88%89-%E1%89%B0%E1%89%A3%E1%88%88&Itemid=212    1
    https://waltainfo.com/am/28427/                                                                                                                                                                                                                                                                                                                                           1
    Name: link, Length: 50008, dtype: int64




```python
data.category.unique()
```




    array(['ሀገር አቀፍ ዜና', 'ፖለቲካ', 'ስፖርት', 'ዓለም አቀፍ ዜና', 'ቢዝነስ', 'መዝናኛ', nan],
          dtype=object)




```python
data['word_len'] = data['article'].str.split().str.len()
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>headline</th>
      <th>category</th>
      <th>date</th>
      <th>views</th>
      <th>article</th>
      <th>link</th>
      <th>word_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15168</th>
      <td>ህወሃት ከፌዴራሊስት ሀይሎች ጋር እየሰራሁ ነው በማለት ህብረተሰቡን እያወ...</td>
      <td>ሀገር አቀፍ ዜና</td>
      <td>26-Sep-20</td>
      <td>2,204</td>
      <td>አዲስ አበባ፣ መስከረም 16፣ 2013 (ኤፍ.ቢ.ሲ) ህወሃት ከፌዴራሊስት ...</td>
      <td>https://www.fanabc.com/%e1%88%85%e1%8b%88%e1%8...</td>
      <td>206</td>
    </tr>
    <tr>
      <th>13721</th>
      <td>አቶ ደመቀ መኮንን ከተመድ የአፍሪካ ቀንድ ልዩ መልእክተኛ ጋር ተወያዩ</td>
      <td>ሀገር አቀፍ ዜና</td>
      <td>1-Dec-20</td>
      <td>336</td>
      <td>አዲስ አበባ፣ ህዳር 22፣ 2013 (ኤፍ.ቢ.ሲ) የኢፌዴሪ ምክትል ጠቅላይ...</td>
      <td>https://www.fanabc.com/%e1%8a%a0%e1%89%b6-%e1%...</td>
      <td>141</td>
    </tr>
    <tr>
      <th>21287</th>
      <td>በአዲስ ዓመት ዋዜማ አራት ሰዎች በማይታወቁ ተሽከርካሪዎች ተገጭተው ሞቱ</td>
      <td>ፖለቲካ</td>
      <td>15-Sep-19</td>
      <td>Unknown</td>
      <td>የ2011 ዓ.ም. ያለ ምንም የወንጀል ድርጊት በሰላም መጠናቀቁን ፖሊስ ቢ...</td>
      <td>https://www.ethiopianreporter.com/article/16750</td>
      <td>206</td>
    </tr>
    <tr>
      <th>16264</th>
      <td>“የህዳሴ ግድቡ በወቅቱ በነበሩ የቦርድ ሰብሳቢ አድራጊ ፈጣሪነት ለከፍተኛ...</td>
      <td>ሀገር አቀፍ ዜና</td>
      <td>18-Jun-20</td>
      <td>1,751</td>
      <td>አዲስ አበባ፣ ሰኔ 11፣ 2012 (ኤፍ. ቢ.ሲ) የህዳሴ ግድብ ግንባታ በ...</td>
      <td>https://www.fanabc.com/%e1%8b%a8%e1%88%85%e1%8...</td>
      <td>261</td>
    </tr>
    <tr>
      <th>24513</th>
      <td>በኦሮሚያ ክልል ንብረቶቻቸው የወደሙባቸው ኢንቨስተሮች የካሳ ጥያቄ እያቀረ...</td>
      <td>ፖለቲካ</td>
      <td>23-Oct-16</td>
      <td>Unknown</td>
      <td>በመስከረም ወር መጨረሻ በኦሮሚያ ክልል በተነሳው ሁከት ንብረቶቻቸው የወደ...</td>
      <td>https://www.ethiopianreporter.com/content/%E1%...</td>
      <td>461</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 50707 entries, 15168 to 15795
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   headline  50694 non-null  object
     1   category  50706 non-null  object
     2   date      50707 non-null  object
     3   views     50707 non-null  object
     4   article   50707 non-null  object
     5   link      50706 non-null  object
     6   word_len  50707 non-null  int64 
    dtypes: int64(1), object(6)
    memory usage: 3.1+ MB



```python
data.word_len.mean()
```




    249.65586605399648



# character level normalization

Amharic has characters wich have the same sound that can be interchangably used.

for example letters 'ሃ','ኅ','ኃ','ሐ','ሓ','ኻ','ሀ' have the same sound so we change them to 'ሀ' 


```python
import re
#method to normalize character level missmatch such as ጸሀይ and ፀሐይ
def normalize_char_level_missmatch(input_token):
    rep1=re.sub('[ሃኅኃሐሓኻ]','ሀ',input_token)
    rep2=re.sub('[ሑኁዅ]','ሁ',rep1)
    rep3=re.sub('[ኂሒኺ]','ሂ',rep2)
    rep4=re.sub('[ኌሔዄ]','ሄ',rep3)
    rep5=re.sub('[ሕኅ]','ህ',rep4)
    rep6=re.sub('[ኆሖኾ]','ሆ',rep5)
    rep7=re.sub('[ሠ]','ሰ',rep6)
    rep8=re.sub('[ሡ]','ሱ',rep7)
    rep9=re.sub('[ሢ]','ሲ',rep8)
    rep10=re.sub('[ሣ]','ሳ',rep9)
    rep11=re.sub('[ሤ]','ሴ',rep10)
    rep12=re.sub('[ሥ]','ስ',rep11)
    rep13=re.sub('[ሦ]','ሶ',rep12)
    rep14=re.sub('[ዓኣዐ]','አ',rep13)
    rep15=re.sub('[ዑ]','ኡ',rep14)
    rep16=re.sub('[ዒ]','ኢ',rep15)
    rep17=re.sub('[ዔ]','ኤ',rep16)
    rep18=re.sub('[ዕ]','እ',rep17)
    rep19=re.sub('[ዖ]','ኦ',rep18)
    rep20=re.sub('[ጸ]','ፀ',rep19)
    rep21=re.sub('[ጹ]','ፁ',rep20)
    rep22=re.sub('[ጺ]','ፂ',rep21)
    rep23=re.sub('[ጻ]','ፃ',rep22)
    rep24=re.sub('[ጼ]','ፄ',rep23)
    rep25=re.sub('[ጽ]','ፅ',rep24)
    rep26=re.sub('[ጾ]','ፆ',rep25)
    #Normalizing words with Labialized Amharic characters such as በልቱዋል or  በልቱአል to  በልቷል  
    rep27=re.sub('(ሉ[ዋአ])','ሏ',rep26)
    rep28=re.sub('(ሙ[ዋአ])','ሟ',rep27)
    rep29=re.sub('(ቱ[ዋአ])','ቷ',rep28)
    rep30=re.sub('(ሩ[ዋአ])','ሯ',rep29)
    rep31=re.sub('(ሱ[ዋአ])','ሷ',rep30)
    rep32=re.sub('(ሹ[ዋአ])','ሿ',rep31)
    rep33=re.sub('(ቁ[ዋአ])','ቋ',rep32)
    rep34=re.sub('(ቡ[ዋአ])','ቧ',rep33)
    rep35=re.sub('(ቹ[ዋአ])','ቿ',rep34)
    rep36=re.sub('(ሁ[ዋአ])','ኋ',rep35)
    rep37=re.sub('(ኑ[ዋአ])','ኗ',rep36)
    rep38=re.sub('(ኙ[ዋአ])','ኟ',rep37)
    rep39=re.sub('(ኩ[ዋአ])','ኳ',rep38)
    rep40=re.sub('(ዙ[ዋአ])','ዟ',rep39)
    rep41=re.sub('(ጉ[ዋአ])','ጓ',rep40)
    rep42=re.sub('(ደ[ዋአ])','ዷ',rep41)
    rep43=re.sub('(ጡ[ዋአ])','ጧ',rep42)
    rep44=re.sub('(ጩ[ዋአ])','ጯ',rep43)
    rep45=re.sub('(ጹ[ዋአ])','ጿ',rep44)
    rep46=re.sub('(ፉ[ዋአ])','ፏ',rep45)
    rep47=re.sub('[ቊ]','ቁ',rep46) #ቁ can be written as ቊ
    rep48=re.sub('[ኵ]','ኩ',rep47) #ኩ can be also written as ኵ  
    return rep48

```


```python
data['article'] = data['article'].str.replace('[^\w\s]','')
```


```python
data['article'] = data['article'].apply(lambda x: normalize_char_level_missmatch(x))
```


```python
n_data = data[['article','category']]
n_data.head()

text,label = data['article'].values,data['category'].values
```


```python
# n_data.head(5).to_csv('table.csv')
```

# Naive Bays - CountVectorizer


```python
from sklearn.feature_extraction.text import CountVectorizer
matrix = CountVectorizer(analyzer='word',max_features=1000,ngram_range=(1, 3))
X = matrix.fit_transform(text).toarray()
X
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [1, 0, 0, ..., 0, 0, 5],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]])




```python
unique_label = list(set(label))
Y= []
for i in label:
    Y.append(unique_label.index(i))
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2)
```


```python
# Naive Bayes 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict Class
y_pred = classifier.predict(X_test)

# Accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

accuracy
```




    0.6220666535200158




```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['ስፖርት', 'መዝናኛ', 'ሀገር አቀፍ ዜና', 'ቢዝነስ', 'ዓለም አቀፍ ዜና', 'ፖለቲካ', 'nan']))
```

                  precision    recall  f1-score   support
    
            ስፖርት       0.82      0.39      0.53      4154
            መዝናኛ       0.37      0.72      0.49       762
      ሀገር አቀፍ ዜና       0.00      0.00      0.00         0
            ቢዝነስ       0.44      0.90      0.59      1334
      ዓለም አቀፍ ዜና       0.96      0.94      0.95      1934
            ፖለቲካ       0.59      0.55      0.57      1808
             nan       0.35      0.78      0.48       150
    
        accuracy                           0.62     10142
       macro avg       0.50      0.61      0.52     10142
    weighted avg       0.72      0.62      0.62     10142
    


    /home/israel/anaconda3/envs/py3.6/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1245: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /home/israel/anaconda3/envs/py3.6/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1245: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /home/israel/anaconda3/envs/py3.6/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1245: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


# Naive Bays - tf -df


```python
from sklearn.feature_extraction.text import TfidfVectorizer
matrix = TfidfVectorizer(analyzer='word',max_features=1000,ngram_range=(1, 3))
X = matrix.fit_transform(text).toarray()
X
```




    array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.06538107, 0.        , 0.        , ..., 0.        , 0.        ,
            0.33719558],
           ...,
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ]])




```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2)
```


```python
# Naive Bayes 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict Class
y_pred = classifier.predict(X_test)

# Accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

accuracy
```




    0.6230526523368172




```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['ስፖርት', 'መዝናኛ', 'ሀገር አቀፍ ዜና', 'ቢዝነስ', 'ዓለም አቀፍ ዜና', 'ፖለቲካ', 'nan']))
```

                  precision    recall  f1-score   support
    
            ስፖርት       0.89      0.34      0.50      4106
            መዝናኛ       0.32      0.82      0.46       754
      ሀገር አቀፍ ዜና       0.00      0.00      0.00         1
            ቢዝነስ       0.62      0.78      0.69      1309
      ዓለም አቀፍ ዜና       0.98      0.95      0.97      1986
            ፖለቲካ       0.49      0.69      0.57      1875
             nan       0.23      0.87      0.36       111
    
        accuracy                           0.62     10142
       macro avg       0.50      0.64      0.51     10142
    weighted avg       0.75      0.62      0.62     10142
    


    /home/israel/anaconda3/envs/py3.6/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1245: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /home/israel/anaconda3/envs/py3.6/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1245: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /home/israel/anaconda3/envs/py3.6/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1245: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))

