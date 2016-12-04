# -*- coding:utf-8 -*-
import json
import os
import urllib2
from google_keys import key_dict

key_dict = key_dict


def url_search(keyword):
    # 画像urlを拾ってくる
    img_urls = []
    base_url = "https://www.googleapis.com/customsearch/v1?key={0}&cx={1}&searchType=image&q={2}&num=10&start={3}"
    # 一回で10個までしか取れないので繰り返し50個取得する。
    for i in range(5):
        start_index = i*10+1
        url = base_url.format(key_dict["api_key"], key_dict["engine_id"], keyword, start_index)
        res = urllib2.urlopen(url)
        data = json.load(res)
        img_urls += [result["link"] for result in data["items"]]

    return img_urls


def url_download(urls):
    # urlから画像をフォルダにdownload
    print("Start Downloading")
    possible_exts = [".png", ".jpg", ".jpeg"]
    img_index = 0
    opener = urllib2.build_opener()
    for url in set(urls):
        try:
            file_name, ext = os.path.splitext(url)
            if ext not in possible_exts:
                continue
            req = urllib2.Request(url, headers={"User-Agent": "Magic Browser"})
            img_file = open('./resources/pictures/mukai/'+str(img_index)+ext, "wb")
            img_file.write(opener.open(req).read())
            img_file.close()
            img_index += 1

        except:
            continue


def main():
    print("mukaiフォルダのリサイズがやり直しになります。つづけますか？ Y/n")
    if raw_input().strip() != "Y":
        sys.exit()
    keyword = "パンサー向井"
    urls = url_search(keyword)
    url_download(urls)
    print("End")

if __name__ == '__main__':
    main()
