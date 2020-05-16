import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from matplotlib import markers,colors
from wordcloud import WordCloud

# PS this code is kinda all over the place. Different parts were ran and commeneted out after generating plots

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
                norm=normClu, label=x[i])
    plt.colorbar(imClu)
    plt.legend().set_draggable(True)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    
def wordcloud(text, max_words):
    '''
    Wrapper around Wordcloud that increases quality, picks a specific font,
    and puts it on a white background
    '''
    
    wordcloud = WordCloud(font_path='C:\WINDOWS\FONTS\PERTILI.TTF',
                          width = 4000,
                          height = 3000,
                          background_color="white",
                          max_words = max_words                          
                         ).generate(text)
    plt.figure(figsize=(40,25))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    return
#give the songs of a decade, returns avg word count per song
def avgWord(songs):
    avg=0
    for lyr in songs['Lyrics']:
        avg += len(lyr)
    avg /= len(songs['Lyrics'])
    return avg
def top_feats_per_cluster(X, y, features, min_tfidf=0.1, top_n=25):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label) 
        feats_df = top_mean_feats(X, features, ids,    min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("cluster = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.score, align='center', color='#7530FF')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.features)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
        plt.show()
#%%
songs = pd.read_csv('all_songs_data.csv',index_col=0)
songs = songs.dropna(subset=['Lyrics'])
songs.reset_index(inplace=True, drop=True) 
stopwords = ENGLISH_STOP_WORDS.union()
vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)
indexNames = songs[ songs['Lyrics'].map(len) > 9000 ].index
songs.drop(indexNames , inplace=True)
songs.reset_index(inplace=True, drop=True) 

d1=songs[0:1048]
d2=songs[1048:2011]
d3=songs[2011:2995]
d4=songs[2995:3960]
d5=songs[3960:4932]
d6=songs[4932:5913]
#decades starting with 1960, ending at 2010
decades= [d1,d2,d3,d4,d5,d6]
recent = songs[4500:]
recent.reset_index(inplace=True, drop=True) 

X = vect.fit_transform(recent['Lyrics'])
features = vect.get_feature_names()
X_dense = X.todense()

top_feats_in_doc(X,features,1,50)
coords = PCA(n_components=2).fit_transform(X_dense)

wordsMean=[]
wordsTop=[]
for d in decades:
    X = vect.fit_transform(d['Lyrics'])
    features = vect.get_feature_names()
    X_dense = X.todense()
    topMeanFeat=top_mean_feats(X, features, top_n=50)
    topFeatures=top_feats_in_doc(X, features, 1, 50)
    wordsMean.append(topMeanFeat)
    wordsTop.append(topFeatures)
    #wordcloud(str(topFeatures),max_words=30)

X = vect.fit_transform(songs[4500:]['Lyrics'])
features = vect.get_feature_names()
X_dense = X.todense()
topFeatures= top_feats_in_doc(X, features, 1, 50)
#wordcloud(str(topFeatures),max_words=30)
tFeat=np.array(topFeatures)
tMeanFeat=np.array(topMeanFeat)
tFeat=np.array(topFeatures)
a=np.flip(tFeat)

plotWithWords(a[:,1],a[:,0])

n_clusters =4 
clf = KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++', n_init=1)
labels = clf.fit_predict(X)

plot_tfidf_classfeats_h(top_feats_per_cluster(X,labels,features))
#%%
from sklearn.cluster import SpectralClustering as sc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
 
cv=CountVectorizer()
word_count_vector=cv.fit_transform(recent['Lyrics'])

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])

df_idf.sort_values(by=['idf_weights'])
count_vector=cv.transform(recent['Lyrics'])
tf_idf_vector=tfidf_transformer.transform(count_vector)

feature_names = cv.get_feature_names()
# (feature_names,tf_idf_vector.data)
#get tfidf vector for first document
first_document_vector=tf_idf_vector[0]
 
#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
z=df.sort_values(by=["tfidf"],ascending=False) 


recent = songs[5870:]
recent.reset_index(inplace=True, drop=True) 

vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.2, min_df=2)
X = vect.fit_transform(recent['Lyrics'])
features = vect.get_feature_names()
X_dense = X.todense()
idk=sc(n_clusters=5,random_state=1,affinity='rbf',n_init=50,gamma=1)
labels = idk.fit_predict(X)
plot_tfidf_classfeats_h(top_feats_per_cluster(X,labels,features))
#top_feats_in_doc(X,features,1,50)

from sklearn.cluster import AgglomerativeClustering as ac
agg= ac(n_clusters=)
labels = agg.fit_predict(X.toarray())
plot_tfidf_classfeats_h(top_feats_per_cluster(X,labels,features))
