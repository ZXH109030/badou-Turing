import matplotlib.pyplot as plt
import  numpy as np
import pandas as pd

reviews = pd.read_csv(r"C:\Users\ZhongXH2\Desktop\zuoye\3-可视化库matpltlib\fandango_scores.csv")
# print(reviews.head())
cols = ['RT_user_norm','Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']
norm_reviews  = reviews[cols]
# print(norm_reviews[:1])

#转换为np.array
bar_height =norm_reviews.loc[0,cols].values


# print(type(norm_reviews.loc[0]))  #仍是serial
# print(type(norm_reviews.loc[0].values)) #ndarray


# print(norm_reviews.loc[1].values)
# print(norm_reviews.loc[2].values)
bar_position = np.arange(5)+0.75
tick_pos=range(1,6)

fig,ax =plt.subplots()
# ax.bar(bar_position,bar_height,0.5)  #竖直柱状图
# ax.barh(bar_position,bar_height,0.5) #横向柱状图

ax.scatter(norm_reviews['Fandango_Ratingvalue'], norm_reviews['RT_user_norm'])
# ax.scatter(norm_reviews.loc[2,cols].values,norm_reviews.loc[1,cols].values)
ax.set_xticks(tick_pos)
ax.set_xticklabels(cols,rotation =45)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("x/y/z")
plt.show()

# fig = plt.figure(figsize=(5,10))
# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(2,1,2)
# ax1.scatter(norm_reviews['Fandango_Ratingvalue'], norm_reviews['RT_user_norm'])
# ax1.set_xlabel('Fandango')
# ax1.set_ylabel('Rotten Tomatoes')
# ax2.scatter(norm_reviews['RT_user_norm'], norm_reviews['Fandango_Ratingvalue'])
# ax2.set_xlabel('Rotten Tomatoes')
# ax2.set_ylabel('Fandango')
# plt.show()

