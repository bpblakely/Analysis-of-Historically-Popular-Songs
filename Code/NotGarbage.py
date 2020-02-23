import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD 
from sklearn.preprocessing import normalize 
from matplotlib import markers,colors
from mpl_toolkits.mplot3d import Axes3D

def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df
def top_feats_in_doc(X, features, row_id, top_n=25):
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)
def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()
        D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

#x = [word,freq] , y = [word,freq]
def plotWithWords(x,y):
    n=x.shape[0]
    markerList = list(markers.MarkerStyle.markers.keys())
    normClu = colors.Normalize(np.min(n),np.max(n))
    for i in range(n):
         
         imClu = plt.scatter(
                x[i], y[i],
                marker=markerList[i % len(markerList)],
                norm=normClu, label=y[i])
    plt.colorbar(imClu)
    plt.legend().set_draggable(True)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
#%%
songs = pd.read_csv('all_songs_data.csv',index_col=0)
songs = songs.dropna(subset=['Lyrics'])
songs.reset_index(inplace=True, drop=True) 
stopwords = ENGLISH_STOP_WORDS.union()
vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)
indexNames = songs[ songs['Lyrics'].map(len) > 9000 ].index
songs.drop(indexNames , inplace=True)
songs.reset_index(inplace=True, drop=True) 

d1=songs[0:1000]
d2=songs[1000:2000]
d3=songs[2000:3000]
d4=songs[3000:4000]
d5=songs[4000:5000]
d6=songs[5000:5913]
#decades starting with 1960, ending at 2010
decades= [d1,d2,d3,d4,d5,d6]
recent = songs[4500:]
recent.reset_index(inplace=True, drop=True) 
X = vect.fit_transform(recent['Lyrics'])
features = vect.get_feature_names()
X_dense = X.todense()
coords = PCA(n_components=2).fit_transform(X_dense)


topMeanFeat=top_mean_feats(X, features, top_n=20)
topFeatures=top_feats_in_doc(X, features, 1, 20)
tMeanFeat=np.array(topMeanFeat)
tFeat=np.array(topFeatures)

tFeat[1,1]

plotWithWords(tFeat[:,1],tFeat[:,0])
tFeat[:,0]
plt.scatter(topFeatures, topMeanFeat, c='m')
plt.show()