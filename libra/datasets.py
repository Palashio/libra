from download import download

import pandas as pd
import os

_dataset_links = {
    'housing': 'https://download1593.mediafire.com/0llwk1so1nwg/bcy99z7mcdcfjin/24824_33185_bundle_archive.zip',
    'fake job postings': 'https://download855.mediafire.com/3emjy3dnq71g/1vaj6fxx465ghyy/533871_976879_bundle_archive+%281%29.zip',
    'landslides': 'https://download947.mediafire.com/p3trn9y69trg/4eto9fdwkj8lay7/686_1296_bundle_archive+%282%29.zip'
}


def load(dataset_name, path='libra_datasets'):

    if not dataset_name.lower() in ['housing', 'fake job postings', 'landslides']:
        raise Exception("Dataset does not exist or is not supported by Libra")

    if dataset_name == 'housing':
        file_name = 'housing.csv'
    elif dataset_name == 'fake job postings':
        file_name = 'fake_job_postings.csv'
    else:
        file_name = 'catalog.csv'

    if not os.path.isfile(path + '/' + file_name):
        download(url=_dataset_links[dataset_name], path=path, kind='zip', progressbar=True, replace=True, timeout=100000)

    return pd.read_csv(path + '/' + file_name)