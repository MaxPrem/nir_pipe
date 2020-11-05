import os
import pandas as pd
from os import listdir
from os.path import isfile, join
import glob

def get_csv_filenames(mypath):
    """ creates list from filenames in drectory 'mypath' """
    file_names = [f for f in listdir(mypath) if f.endswith('.csv') if isfile(join(mypath, f))]
    file_names = [os.path.splitext(each)[0] for each in file_names]
    return file_names

def import_spectra_to_dataframe(mypath):
    """ concatenate single spectra to dataframe containing all meassurements """
    mypath = mypath + '*.csv'
    files = glob.glob(mypath)
    df = pd.concat([pd.read_csv(fp, index_col=None,skiprows=2, usecols=[1]) for fp in files], axis=1)
    df = df.T
    return df


def ceate_named_spectra_df(mypath, wavenumbers):

    file_names = get_csv_filenames(mypath)
    spectra = import_spectra_to_dataframe(mypath)
    spectra.columns = wavenumbers
    spectra.index = file_names
    return spectra


if __name__ == '__main__':


    # set wavenumbers
    wavenumbers = [i for i in range(13000, 4000, -2)]
    # all csv files in given directory

    # path to filenames file names
    mypath= 'riniExport/riniCSV/'
    # use function to create dataframe with col und row names
    spectra = ceate_named_spectra_df(mypath, wavenumbers)
    # show head of dataframe
    spectra.head()
