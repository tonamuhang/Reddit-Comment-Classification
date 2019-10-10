from bs4 import BeautifulSoup
from urllib.request import urlopen
from urlextract import URLExtract

string = "here is my website: (https://www.youtube.com/watch?v=VnCYftlM-zg"

extractor = URLExtract()
for url in extractor.gen_urls(string):
    text = urlopen(url).read()
    text = BeautifulSoup(text, features="html.parser").find('title').string  # HTML decoding
    string += text


print(string)