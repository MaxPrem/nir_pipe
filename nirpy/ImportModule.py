import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_spec(wave_number, X):
    """plots spectra by setting the wavenumber to the x-achsis and samples absorption values to y-achsis"""
    fig, ax = plt.subplots()
    with plt.style.context(("ggplot")):
        plt.plot(wave_number, X.T)
        ax.set_xlim(wave_number.max(), wave_number.min())  # decreasing
        ax.set_xlabel("Wavenumber (cm-1)")
        ax.set_ylabel("Absorbance spectra")
        ax.grid(True)

        plt.show()
        print("Spectra rangeing from", wave_number.max(), "to", wave_number.min(), "cm-1")

def cut_specs(specs, start, stop, plot=True):
    """Cuts spectra by columnname"""

    specs = specs.set_index("Unnamed: 0").sort_index()

    if start < stop:
        stop, start = start, stop

    X = specs.loc[:, str(start) : str(stop)]

    wave_number = X.columns.astype(int)

    if plot == True:
        plot_spec(wave_number,X)

    return specs.loc[:, str(start) : str(stop)]


# class DataLoader(object):
#     """Spectra and Labvalues"""
#     def __init__(self, spectra, lab_values, lab_col=1):
#         self.spectra = spectra
#         self.lab_values = lab_values
#         self.lab_col = lab_col
#     def load_spectra():
#         spectra =  pd.read_csv('./luzrawSpectra/nirMatrix.csv')
#         return spectra
#     def load_lab_values():
#


def importLuzCol(specs, lab, lab_col=4):

    # sort by samplenames if not already done by cut cut_specs
    # maybe changeing import to a class.....
    if specs.columns[0] == "Unnamed: 0":
        specs = specs.set_index("Unnamed: 0").sort_index()
    else:
        print("Reload specs from file.")
    wave_number = specs.columns.astype(int)
    lab = lab.set_index(["Code"]).sort_index()
    # wave_number.shape
    # lab.shape
    # specs

    # remove missing values
    """insert outlier removal"""
    idx1 = pd.Index(lab.index)
    idx2 = pd.Index(specs.index)

    remSpec = idx1.difference(idx2).values
    remSpec
    lab = lab.drop(remSpec)

    remSpec = idx2.difference(idx1).values
    remSpec
    specs = specs.drop(remSpec)

    # set lab values
    lab_values = [
        c
        for c in lab.columns.values
        if c
        not in ["Code", "Sorte", "Jahr", "Aufwuchs", "Sorte.1", "Beobachtung", "Stück"]
    ]

    lab_class = [
        c
        for c in lab.columns.values
        if c in ["Sorte", "Jahr", "Aufwuchs", "Sorte.1", "Beobachtung", "Stück"]
    ]
    # Set Variables for Regression
    X = specs.values
    y = lab[lab_values[lab_col]].values

    print("Features for Regressoin:", lab_values)
    print("Features for Classification:", lab_class)
    print("Setting y to", lab_values[lab_col])

    ref = lab_values[lab_col]

    return X, y, wave_number, ref


def sel_wavs(specs, lab, sel_feats, lab_col=4):

    """"""

    # sort by samplenames

    specs = specs.set_index("Unnamed: 0").sort_index()
    specs = specs[sel_feats]
    wave_number = specs.columns.astype(int)
    lab = lab.set_index(["Code"]).sort_index()
    # wave_number.shape
    # lab.shape
    # specs

    # remove missing values
    """insert outlier removal"""
    idx1 = pd.Index(lab.index)
    idx2 = pd.Index(specs.index)

    remSpec = idx1.difference(idx2).values
    remSpec
    lab = lab.drop(remSpec)

    remSpec = idx2.difference(idx1).values
    remSpec
    specs = specs.drop(remSpec)

    # set lab values
    lab_values = [
        c
        for c in lab.columns.values
        if c
        not in ["Code", "Sorte", "Jahr", "Aufwuchs", "Sorte.1", "Beobachtung", "Stück"]
    ]

    lab_class = [
        c
        for c in lab.columns.values
        if c in ["Sorte", "Jahr", "Aufwuchs", "Sorte.1", "Beobachtung", "Stück"]
    ]
    # Set Variables for Regression
    X = specs.values
    y = lab[lab_values[lab_col]].values

    print("Features for Regressoin:", lab_values)
    print("Features for Classification:", lab_class)
    print("Setting y to", lab_values[lab_col])
    return X, y, wave_number


if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt

    # path
    specs = pd.read_csv("./luzrawSpectra/nirMatrix.csv")

    lab = pd.read_excel("./luzrawSpectra/labdata_named.xlsx")

    specs = pd.read_csv("/Users/maxprem/nirPy/calData_full.csv")
    #
    # ss = specs.loc[:,str(5000):str(4000)]
    # ss

    specs.set_index(1)
    cut_specs = cut_specs(specs, 5000, 4000)
    cut_specs2 = cut_specs(specs, 5000, 4000)

    specs.index
    cut_specs.index

    X, y, wave_number = importLuzCol(cut_specs, lab, 4)
    X

    _ = plt.plot(-wave_number, X.T,)

    specs.set_index(list(specs.columns[0]))

    specs.iloc[:, 0].set_index()

    #
#
#
#
# X, y , wave_number, regressors, classes = importLuzCol(specs, lab, lab_col=4)
# X.shape
# y.shape
#
# from sklearn.model_selection import  train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
