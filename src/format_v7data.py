import sys
from os import listdir, rename
import json
import requests


def format_data(PATH, download=False):
    """Modifies the names of JSON files downloaded from Darwin v7 in place,
    and downloads the original images (if specified), places these in the
    same directory. Names are modified to match original names of images
    when uploaded to v7.

    Arguments:
        PATH {string} -- path to the unzipped folder downloaded directly
                         from Darwin v7.
    Keyword Arguments:
        download {bool} -- Option to download images from v7 hosting.
    """
    # iterate over files in directory
    for file_name in listdir(PATH):
        if file_name[0] == '.':  # skip .DS_Store and other hidden files
            continue
        # open each JSON file
        with open(PATH+'/'+file_name) as json_file:
            data = json.load(json_file)

            # get image url and download it
            image_name = data['image']['original_filename']
            if download:
                url = data['image']['url']
                download_file_from_url(url, image_name, PATH)

        # rename json file to match image name
        hash_name = image_name[:-4]
        rename(PATH+'/'+file_name, PATH+'/'+hash_name+'.json')


def download_file_from_url(url, PATH, file_name):
    """Takes a URL and downloads file at the location,
    places it in the directory specified with name specified.

    Arguments:
        url {string} -- full URL address of file download
        PATH {string} -- path to the desired download location
        file_name {string} -- name given downloaded file.
    """
    with requests.get(url) as r:
        with open(PATH+'/'+file_name, 'wb') as f:
            f.write(r.content)


if __name__ == "__main__":
    PATH = sys.argv[1]
    format_data(PATH)
