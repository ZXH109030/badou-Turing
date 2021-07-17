import pandas as pd
import matplotlib.pyplot as plt
reviews = pd.read_csv(r"C:\Users\ZhongXH2\Desktop\zuoye\3-可视化库matpltlib\fandango_scores.csv")
cols = ['FILM', 'RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue']
norm_reviews = reviews[cols]
print(norm_reviews[:1])


#计算数值个数value_counts()
fandango_distribution = norm_reviews['Fandango_Ratingvalue'].value_counts()
fandango_distribution = fandango_distribution.sort_index()

imdb_distribution = norm_reviews['IMDB_norm'].value_counts()
imdb_distribution = imdb_distribution.sort_index()

print(fandango_distribution)
print(imdb_distribution)

#设定bins=20在X轴上的分段数20， 显示区间范围range=(4, 5)
# fig,ax =plt.subplots()
# # # ax.hist(norm_reviews['Fandango_Ratingvalue'])
# ax.hist(norm_reviews['Fandango_Ratingvalue'],bins=5)
# # ax.hist(norm_reviews['Fandango_Ratingvalue'], range=(4, 5),bins=20)
# plt.show()

fig  = plt.figure(figsize=(5,10))
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)

ax1.hist(norm_reviews['Fandango_Ratingvalue'],bins=20,range=(0,5))
ax1.set_title('Distribution of Fandango Ratin')
ax1.set_ylim(0,50)
ax1.set_xlim(0,6)
ax2.hist(norm_reviews['RT_user_norm'], 20, range=(0, 5))
ax2.set_title('Distribution of Rotten Tomatoes Ratings')
ax2.set_ylim(0, 50)

ax3.hist(norm_reviews['Metacritic_user_nom'], 20, range=(0, 5))
ax3.set_title('Distribution of Metacritic Ratings')
ax3.set_ylim(0, 50)

ax4.hist(norm_reviews['IMDB_norm'], 20, range=(0, 5))
ax4.set_title('Distribution of IMDB Ratings')
ax4.set_ylim(0, 50)
plt.show()

# num_cols = ['RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue']
# fig, ax = plt.subplots()
# # ax.boxplot(norm_reviews['RT_user_norm'])
# # ax.set_xticklabels(['Rotten Tomatoes'])
# ax.boxplot(reviews[num_cols].values)
# ax.set_xticklabels(num_cols, rotation=90)
# ax.set_ylim(0, 5)
# plt.show()