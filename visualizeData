import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('reddit_train.csv')
print(df.head())
fig = plt.figure(figsize=(10,8))
df.groupby('subreddits').comments.count().plot.bar(ylim=0)
plt.show()
# plot the data
# found 3500 samples * 20 classes = 70k samples in total
# all class samples are balanced