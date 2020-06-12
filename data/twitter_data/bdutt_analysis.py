import pandas as pd
from tensorflow.keras.models import Sequential

df =pd.read_csv('bdutt.csv'  )
sorted_df = df.sort_values(['1'], ascending=False)
print(sorted_df.iloc[0:100])

model = Sequential()
model.fit(validation_split=.2)