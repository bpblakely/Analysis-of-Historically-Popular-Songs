import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import lyricsgenius as genius
import sys
import re
from datetime import datetime
pd.options.mode.chained_assignment = None
sys.path.insert(1, "F:\Python File Saves")

# This code is almost entirely from https://towardsdatascience.com/49-years-of-lyrics-why-so-angry-1adf0a3fa2b4 
# In this post, the author built a scrapper that does what I want to do. The main website being used to get ranks died, so I use the time machine for it


def collect_songs_from_billboard(start_year,end_year):
    '''This function takes in a start year and and end year, then iterates through each year to 
    pull song data from billboard or bobborst as needed. Then it uses beautiful soup to clean
    the data. Finally it stores the cleaned data in a dataframe and returns it
    
    Parameters:
    
    start_year (int): the year to start at.
    end_year (int): the year to end at.
    Returns: 
    
    dataframe.
    '''
    
    years = np.arange(start_year, end_year + 1).astype(int)
    dataset = pd.DataFrame()
    url_list = []
    ### Billboard doesn't have it's own complete results from 1970 to 2016,
    ### so we'll use bobborst.com as our primary and collect from billboard as needed
    alternate_site_collection_range = np.arange(start_year, 2017)
    #URL Constructor
    for i in range (0, len(years)):
        url_list.append("https://www.billboard.com/charts/year-end/" + str(years[i]) + "/hot-100-songs")      
    for i in range(0, len(url_list)):
        if years[i] in alternate_site_collection_range:
            sys.stdout.write("\r" + "Collecting Songs from " +str(years[i]) + " via http://www.bobborst.com")
            sys.stdout.flush()    
            url = "https://web.archive.org/web/20190816053139/http://www.bobborst.com/popculture/top-100-songs-of-the-year/?year=" + str(years[i])
            page = requests.get(url)
            soup = BeautifulSoup(page.content, "html.parser")
            table = soup.find('table', {'class': 'sortable alternate songtable'})
            rows = table.find_all('tr')
            for j in range(2,102):
                columns = rows[j].find_all('td')
                #print(columns)
                row = {
                    "Rank": columns[0].get_text(strip=True),
                    "Artist": columns[1].get_text(strip=True),
                    "Song Title": columns[2].get_text(strip=True),
                    "Year": years[i]
                }
                dataset = dataset.append(row, ignore_index=True)
            
        else:
            sys.stdout.write("\r" + "Collecting Songs from " +str(years[i]) + " via https://www.billboard.com")
            sys.stdout.flush()
            url = "https://www.billboard.com/charts/year-end/" + str(years[i]) + "/hot-100-songs"
            page = requests.get(url)
            soup = BeautifulSoup(page.content, "html.parser")
            all_ranks = soup.find_all("div", class_="ye-chart-item__rank")
            all_titles = soup.find_all('div', class_="ye-chart-item__title")
            all_artists = soup.find_all("div", class_="ye-chart-item__artist")
            for j in range (0, len(all_ranks)):
                row = {
                    "Rank": all_ranks[j].get_text(strip=True),
                    "Song Title": all_titles[j].get_text(strip=True),
                    "Artist": all_artists[j].get_text(strip=True),
                    "Year": years[i]
                }
                dataset = dataset.append(row, ignore_index=True)
    dataset['Year'] = dataset['Year'].astype(int)
    return dataset

all_songs = collect_songs_from_billboard(1959,2019)

# Now we have all the song ranks, titles, artists, and years. Need to get the lyrics next

# Getting lyrics by useing Genius Lyrics' API
api = genius.Genius("GET YOUR OWN API KEY",sleep_time=0.01, verbose=False)
all_song_data = pd.DataFrame()
start_time = datetime.now()
print("Started at {}".format(start_time))
for i in range(0, len(all_songs)):
    rolling_pct = int((i/len(all_songs))*100)
    print(str(rolling_pct) + "% complete." + " Collecting Record " + str(i) +" of " +
          str(len(all_songs)) +". Year " + str(all_songs.iloc[i]['Year']) + "." + " Currently collecting " + 
          all_songs.iloc[i]['Song Title'] + " by " + all_songs.iloc[i]['Artist'] + " "*50, end="\r")
    song_title = all_songs.iloc[i]['Song Title']
    song_title = re.sub(" and ", " & ", song_title)
    artist_name = all_songs.iloc[i]['Artist']
    artist_name = re.sub(" and ", " & ", artist_name)

    try:
        song = api.search_song(song_title, artist=artist_name)
        #print(song)
        song_album = song.album
        song_album_url = song.album_url
        featured_artists = song.featured_artists
        song_lyrics = re.sub("\n", " ", song.lyrics) #Remove newline breaks, we won't need them.
        song_media = song.media
        song_url = song.url
        song_writer_artists = song.writer_artists
        song_year = song.year
    except:
        song_album = "null"
        song_album_url = "null"
        featured_artists = "null"
        song_lyrics = "null"
        song_media = "null"
        song_url = "null"
        song_writer_artists = "null"
        song_year = "null"
        
    row = {
        "Year": all_songs.iloc[i]['Year'],
        "Rank": all_songs.iloc[i]['Rank'],
        "Song Title": all_songs.iloc[i]['Song Title'],
        "Artist": all_songs.iloc[i]['Artist'],
        "Album": song_album,
        "Album URL": song_album_url,
        "Featured Artists": featured_artists,
        "Lyrics": song_lyrics,
        "Media": song_media,
        "Song URL": song_url,
        "Writers": song_writer_artists,
        "Release Date": song_year
    }
    all_song_data = all_song_data.append(row, ignore_index=True)
end_time = datetime.now()
print("\nCompleted at {}".format(start_time))
print("Total time to collect: {}".format(end_time - start_time))
all_song_data.to_csv("all_songs_data.csv")
all_song_data.to_json("all_song_data.json", orient='records')

