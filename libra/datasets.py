from download import download

import pandas as pd
import os

_dataset_links = {
    'housing': 'https://download1325.mediafire.com/5sm8nmw2gixg/x8m5sol30wz5kjq/5227_7876_bundle_archive+%282%29.zip',
    'fake job postings': 'https://download855.mediafire.com/kezk0rq1ogzg/ikvbeoyirm92qpf/533871_976879_bundle_archive.zip',
    'landslides': 'https://download947.mediafire.com/0mgw8yaubcjg/yd2b09ty4qsk6qb/686_1296_bundle_archive+%281%29.zip'
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
