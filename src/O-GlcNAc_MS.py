#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
import urllib.request
from random import randint
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score



'''
mass spec data with O-GlcNAc sites annotated taken from Burlingame paper:
Global Identification and Characterization of Both O-GlcNAcylation and Phosphorylation at the Murine Synapse
Trinidad et al. 2012 Mol Cell Proteomics
http://www.mcponline.org/content/11/8/215/suppl/DC1
'''

"""Read spreadsheet in as a pandas dataframe"""

# file = "C:\Users\konzmandw.NIH\Documents\MachineLearning\Burlingame2012_MSdata_dwk.csv"
path = "/Users/brian.konzman/PycharmProjects/dan_ml_project/data/ms-data.tsv"
data = pd.read_csv(path, delimiter='\t', header=0)

data.rename(columns={'Unnamed: 17': 'Site1_in_peptide'}, inplace=True)
data.rename(columns={'Unnamed: 18': 'Site2_in_peptide'}, inplace=True)
data.rename(columns={'Unnamed: 19': 'Site3_in_peptide'}, inplace=True)

# print (data['Site1_in_peptide'])


'''
The format of the data is in a very cumbersome format, so this code was written to parse it into a more useable form.
The site assignments refer to the location within a peptide, not the full protein, and look like this: HexNAc@7=7;Oxidation@8=14;HexNAc@21|23
Some site calls are uncertain between several S/T residues on the peptide, so I only take unambiguous calls for further analysis.

'''

'''Here I iterate through the dataframe, pulling out site# from confidant site assignments
these are stored as a string (multiple sites separated by commas) in a dictionary with the row index as a key'''
sites = {}
allsites = {}
for index, row in data.iterrows():
    if "=" in row['Site_Assignment']:
        for j in row['Site_Assignment'].split("=")[:-1]:
            if "HexNAc" not in j:
                continue
            if "@" in j[-2:]:
                if index in sites:
                    sites[index] += ',' + j[-1]
                    allsites[index] += ',' + j[-1]
                else:
                    sites[index] = j[-1]
                    allsites[index] = j[-1]
            else:
                if index in sites:
                    sites[index] += ',' + j[-2:]
                    allsites[index] += ',' + j[-2:]
                else:
                    sites[index] = j[-2:]
                    allsites[index] = j[-2:]
    if "|" in row['Site_Assignment']:
        for j in row['Site_Assignment'].split("|"):
            if "&" in j:
                for k in j.split("&"):
                    if "@" in k[-2:]:
                        if index in allsites:
                            allsites[index] += ',' + k[-1]
                        else:
                            allsites[index] = k[-1]
                    else:
                        if index in allsites:
                            allsites[index] += ',' + k[-2:]
                        else:
                            allsites[index] = k[-2:]
            else:
                if "@" in j[-2:]:
                    if index in allsites:
                        allsites[index] += ',' + j[-1]
                    else:
                        allsites[index] = j[-1]
                else:
                    if index in allsites:
                        allsites[index] += ',' + j[-2:]
                    else:
                        allsites[index] = j[-2:]

'''Here I add the site numbers into the dataframe'''
indices, resnum = zip(*sites.items())
data.loc[indices, ['Site1_in_peptide']] = resnum
indices, resnum = zip(*allsites.items())
data.loc[indices, ['Site2_in_peptide']] = resnum

'''Now I make a new dictionary from the dataframe, storing uniprot# as key, with a list of modified sites as the value
Many proteins have multiple sites and occupy seperate rows in the dataframe - this step unites all sites with one protein
This step also adds the site within the peptide to the start site to assign the site to the residue number of the full-length protein'''
oglcnac_site = {}
for index, row in data.iterrows():
    if type(row['Site1_in_peptide']) == str:
        s = row['Site1_in_peptide'].split(",")
        for i in s:
            i = int(i)
            # print (i + row['Start_Residue'])
            if row['Uniprot_Accession'] in oglcnac_site:
                if i + row['Start_Residue'] in oglcnac_site[row['Uniprot_Accession']]:  # avoid redundant sites
                    continue
                else:
                    oglcnac_site[row['Uniprot_Accession']].append(i + row['Start_Residue'])
            else:
                oglcnac_site[row['Uniprot_Accession']] = [i + row['Start_Residue']]

allpossible_site = {}
for index, row in data.iterrows():
    if type(row['Site2_in_peptide']) == str:
        s = row['Site2_in_peptide'].split(",")
        for i in s:
            i = int(i)
            if row['Uniprot_Accession'] in allpossible_site:
                if i + row['Start_Residue'] in allpossible_site[row['Uniprot_Accession']]:  # avoid redundant sites
                    continue
                else:
                    allpossible_site[row['Uniprot_Accession']].append(i + row['Start_Residue'])
            else:
                allpossible_site[row['Uniprot_Accession']] = [i + row['Start_Residue']]
print(len(oglcnac_site.keys()))
print(len(allpossible_site.keys()))
for key, item in allpossible_site.items():
    print(key, len(item), item)

# allprots = ''
# for key in oglcnac_site.keys():
#    allprots+=key+' '
# print (allprots)

# format: https://www.uniprot.org/uniprot/P12345.fasta




'''Generate +/-5 sequence and for each residue from fasta seq on uniprot website'''


flankingseq = {}
fullseq = {}
for key, value in oglcnac_site.items():
    url = 'https://www.uniprot.org/uniprot/' + key + '.fasta'
    contents = str(urllib.request.urlopen(url).read())
    sequence = ""
    for line in contents.split('\n')[0].split('\\n'):
        if ">" in line:
            continue
        if "'" in line:
            continue
        else:
            sequence += line
    fullseq[key] = sequence
    for i in value:
        if len(sequence) < i + 3:  # some sites were located at an index greater than the sequence length and thus excluded
            continue
        elif sequence[
            (i - 2)] == "S":  # if it isn't S/T, it has been mapped incorrectly, or another type of glycosylation
            # print (key, i, sequence[(i-2)])
            if key in flankingseq:
                flankingseq[key].append(sequence[(i - 7):(i + 4)])
            else:
                flankingseq[key] = [sequence[(i - 7):(i + 4)]]
        elif sequence[(i - 2)] == "T":
            # print (key, i, sequence[(i-2)])
            if key in flankingseq:
                flankingseq[key].append(sequence[(i - 7):(i + 4)])
            else:
                flankingseq[key] = [sequence[(i - 7):(i + 4)]]
        else:
            continue

"""Remove sequences that are not 11mers, that may arise from the termini of the sequence"""
for key, value in flankingseq.items():
    flankingseq[key] = [item for item in flankingseq[key] if len(item) == 11]


'''Toy list to troubleshoot on'''
l = ['HexNAc@3=8',
     'HexNAc@4=19',
     'HexNAc@1=74',
     'HexNAc@16|17',
     'HexNAc@13|17',
     'HexNAc@13|14|15',
     'HexNAc@8=8;HexNAc@13=8',
     'HexNAc@7|10',
     'HexNAc@8=7',
     'HexNAc@9=7;HexNAc@11=7',
     'HexNAc@8=11;HexNAc@9=22;Oxidation@16',
     'HexNAc@9=16;HexNAc@7|8',
     'HexNAc@10=7',
     'HexNAc@7=7;Oxidation@8=14;HexNAc@21|23',
     'HexNAc@8=13;HexNAc@13|14',
     'HexNAc@1&11|1&12|1&14|2&11|2&12|2&14|3&14'
     ]

'''write a fasta file containing all proteins included in analysis
used in netsurfp to predict solvent accessibility and secondary structure'''
fasta = ""
for key, value in fullseq.items():
    fasta += ">" + key + "\n" + value + "\n"
print(fasta)

f = open("modifiedproteins.fasta", "w")
f.write(fasta)
f.close()



"""Generate control dataset, random sites in the same proteins not within 10 residues of a possible O-GlcNAc site"""


ctrl_site = {}
ctrl_flankingseq = {}
for key, item in flankingseq.items():
    allpos_flank = []
    for site in allpossible_site[key]:  # this creates a list of site numbers, plus the numbers of sites +/-10
        for x in range(site - 10, site + 11):
            allpos_flank.append(x)
    randsites = []
    # print (item)
    while len(randsites) < len(
            item):  # this creates a random number, checks if it is near a possible site, then adds to control set
        randpos = randint(5, (len(fullseq[key]) - 5))
        if randpos not in allpos_flank:
            randsites.append(randpos)
    ctrl_site[key] = randsites
    for i in randsites:
        if key in ctrl_flankingseq:
            ctrl_flankingseq[key].append(fullseq[key][(i - 7):(i + 4)])
        else:
            ctrl_flankingseq[key] = [fullseq[key][(i - 7):(i + 4)]]
for key, value in ctrl_flankingseq.items():
    ctrl_flankingseq[key] = [item for item in ctrl_flankingseq[key] if len(item) == 11]

# for key, item in ctrl_flankingseq.items():
# print (key, item)

# for key, item in flankingseq2.items():
# print (key, item)




"""NetSurfP output from all the protein sequences was saved as a txt file, here is read into a new pandas dataframe
The columns are not labeled in the file, so I manually add them"""



path = "NetSurfP_Burlingame.txt"
netsurfp = pd.read_csv(path, delim_whitespace=True, header=None,
                       names=["Exposed", "aa", "Uniprot", "aa#", "RSA", "ASA", "RSA_zscore", "p(a-helix)",
                              "p(b-strand)", "p(coil)"])
# netsurfp.head()




"""Several dictionaries and lists that describe physical features of amino acids"""



aanames = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

hbond_donors = ["R", "N", "Q", "H", "L", "S", "T", "W", "Y"]

hbond_acceptors = ["N", "D", "Q", "E", "H", "S", "T", "Y"]

aromatic = ["F", "W", "Y"]

pKa = {'D': 3.9,
       'E': 4.3,
       'H': 6.1,
       'C': 8.3,
       'Y': 10.1,
       'K': 10.5,
       'R': 12}

# hydrophobicity
octanol = {
    "A": -0.50, "C": 0.02, "D": -3.64, "E": -3.63,
    "F": 1.71, "G": -1.15, "H": -0.11, "I": 1.12,
    "K": -2.80, "L": 1.25, "M": 0.67, "N": -0.85,
    "P": -0.14, "Q": -0.77, "R": -1.81, "S": -0.46,
    "T": -0.25, "V": 0.46, "W": 2.09, "Y": 0.71, }

# Flexibility
# Normalized flexibility parameters (B-values), average 
# Vihinen M., Torkkila E., Riikonen P. Proteins. 19(2):141-9(1994). 
Flex = {'A': 0.984, 'C': 0.906, 'E': 1.094, 'D': 1.068,
        'G': 1.031, 'F': 0.915, 'I': 0.927, 'H': 0.950,
        'K': 1.102, 'M': 0.952, 'L': 0.935, 'N': 1.048,
        'Q': 1.037, 'P': 1.049, 'S': 1.046, 'R': 1.008,
        'T': 0.997, 'W': 0.904, 'V': 0.931, 'Y': 0.929}

# Hydrophilicity 
# 1 Hopp & Wood 
# Proc. Natl. Acad. Sci. U.S.A. 78:3824-3828(1981). 
hw = {'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0,
      'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8,
      'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0,
      'S': 0.3, 'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5}

# Surface accessibility 
# Vergoten G & Theophanides T, Biomolecular Structure and Dynamics, 
# pg.138 (1997). 
# 1 Emini Surface fractional probability 
em = {'A': 0.815, 'R': 1.475, 'N': 1.296, 'D': 1.283, 'C': 0.394,
      'Q': 1.348, 'E': 1.445, 'G': 0.714, 'H': 1.180, 'I': 0.603,
      'L': 0.603, 'K': 1.545, 'M': 0.714, 'F': 0.695, 'P': 1.236,
      'S': 1.115, 'T': 1.184, 'W': 0.808, 'Y': 1.089, 'V': 0.606}

# 2 Janin Interior to surface transfer energy scale 
ja = {'A': 0.28, 'R': -1.14, 'N': -0.55, 'D': -0.52, 'C': 0.97,
      'Q': -0.69, 'E': -1.01, 'G': 0.43, 'H': -0.31, 'I': 0.60,
      'L': 0.60, 'K': -1.62, 'M': 0.43, 'F': 0.46, 'P': -0.42,
      'S': -0.19, 'T': -0.32, 'W': 0.29, 'Y': -0.15, 'V': 0.60}

# VarianceThreshold




"""Generate a matrix of data about the physical features near the O-GlcNAc or random control sites"""

size = 0
for value in flankingseq.values():
    size += len(value)
# size = 964

fullmatrix = np.zeros(((size * 2), (39 * 11)))  # a row for each true datapoint and each control datapoint
# fullmatrix.shape = (964, 429)
# 964 datapoints (sites), 11 aas per site with 39 features each

"""generating features for true data and adding to matrix"""
count = 0
for key, value, in flankingseq.items():
    nsp = netsurfp[
        netsurfp['Uniprot'] == key]  # create new dataframe from netsurfp containing only entries for one protein
    if len(value) < 1:
        continue
    if len(value) != len(oglcnac_site[key]):
        continue
    if key not in oglcnac_site:
        continue
    for site in zip(oglcnac_site[key], value):
        matrix = np.zeros((39, 11))
        for index, aa in enumerate(site[1]):
            np.eye(20)[aanames.index(aa)]
            matrix[0:20, index]
            matrix[0:20, index] = np.eye(20)[aanames.index(aa)]  # aa identity
            matrix[20, index] = octanol[aa]  # hydrophobicity
            matrix[21, index] = float(octanol[aa] > 0)  # is hydrophobic
            matrix[22, index] = Flex[aa]  # flexibility
            matrix[23, index] = hw[aa]  # hydrophobicity
            matrix[24, index] = float(hw[aa] > 0)  # is hydrophilic
            matrix[25, index] = em[aa]  # surface accessibility
            matrix[26, index] = ja[aa]  # surface accessibility
            matrix[27, index] = pKa.get(aa, 7.0)  # pKa
            matrix[28, index] = float(pKa.get(aa, 7.0) > 7)  # is basic
            matrix[29, index] = float(pKa.get(aa, 7.0) < 7)  # is acidic
            matrix[30, index] = float(nsp['Exposed'].tolist()[site[0] - 2 - 5 + index] == "E")  # is exposed
            matrix[31, index] = nsp['RSA'].tolist()[site[0] - 2 - 5 + index]  # RSA score
            matrix[32, index] = nsp['ASA'].tolist()[site[0] - 2 - 5 + index]  # ASA score
            matrix[33, index] = nsp['p(a-helix)'].tolist()[site[0] - 2 - 5 + index]  # p(a-helix)
            matrix[34, index] = nsp['p(b-strand)'].tolist()[site[0] - 2 - 5 + index]  # p(b-strand)
            matrix[35, index] = nsp['p(coil)'].tolist()[site[0] - 2 - 5 + index]  # p(coil)
            matrix[36, index] = float(aa in hbond_donors)  # is H-bond donor
            matrix[37, index] = float(aa in hbond_acceptors)  # is H-bond acceptor
            matrix[38, index] = float(aa in aromatic)  # is aromatic
        fullmatrix[count] = matrix.flatten("F")
        count += 1

"""generating features for control data and adding to matrix"""
for key, value, in ctrl_flankingseq.items():
    nsp = netsurfp[netsurfp['Uniprot'] == key]
    if len(value) < 1:
        continue
    if len(value) != len(ctrl_site[key]):
        continue
    if key not in ctrl_site:
        continue
    for site in zip(ctrl_site[key], value):
        matrix = np.zeros((39, 11))
        for index, aa in enumerate(site[1]):
            np.eye(20)[aanames.index(aa)]
            matrix[0:20, index]
            matrix[0:20, index] = np.eye(20)[aanames.index(aa)]  # aa identity
            matrix[20, index] = octanol[aa]  # hydrophobicity
            matrix[21, index] = float(octanol[aa] > 0)  # is hydrophobic
            matrix[22, index] = Flex[aa]  # flexibility
            matrix[23, index] = hw[aa]  # hydrophobicity
            matrix[24, index] = float(hw[aa] > 0)  # is hydrophilic
            matrix[25, index] = em[aa]  # surface accessibility
            matrix[26, index] = ja[aa]  # surface accessibility
            matrix[27, index] = pKa.get(aa, 7.0)  # pKa
            matrix[28, index] = float(pKa.get(aa, 7.0) > 7)  # is basic
            matrix[29, index] = float(pKa.get(aa, 7.0) < 7)  # is acidic
            matrix[30, index] = float(nsp['Exposed'].tolist()[site[0] - 2 - 5 + index] == "E")  # is exposed netsurfp
            matrix[31, index] = nsp['RSA'].tolist()[site[0] - 2 - 5 + index]  # RSA score
            matrix[32, index] = nsp['ASA'].tolist()[site[0] - 2 - 5 + index]  # ASA score
            matrix[33, index] = nsp['p(a-helix)'].tolist()[site[0] - 2 - 5 + index]  # p(a-helix)
            matrix[34, index] = nsp['p(b-strand)'].tolist()[site[0] - 2 - 5 + index]  # p(b-strand)
            matrix[35, index] = nsp['p(coil)'].tolist()[site[0] - 2 - 5 + index]  # p(coil)
            matrix[36, index] = float(aa in hbond_donors)  # is H-bond donor
            matrix[37, index] = float(aa in hbond_acceptors)  # is H-bond acceptor
            matrix[38, index] = float(aa in aromatic)  # is aromatic
        fullmatrix[count] = matrix.flatten("F")
        count += 1



"""Create y, a vector of the true labels for the sites.  1=O-GlcNAc, 0=not"""
print(fullmatrix.shape)
# 1928 rows(individual data points), 429 columns (features)
y = []
for x in range(964):
    y.append(1)
for x in range(964):
    y.append(0)
print(len(y))

"""list of names of the 429 features (39 features per aa in 11mer)"""
feature_names = []
for i in range(-5, 6):
    for j in aanames:
        feature_names.append(str(i) + j)
    feature_names.append(str(i) + 'hydrophobicity')
    feature_names.append(str(i) + 'is hydrophobic')
    feature_names.append(str(i) + 'flexibility')
    feature_names.append(str(i) + 'hydrophobicity')
    feature_names.append(str(i) + 'is hydrophilic')
    feature_names.append(str(i) + 'surface accessibility')
    feature_names.append(str(i) + 'surface accessibility')
    feature_names.append(str(i) + 'pKa')
    feature_names.append(str(i) + 'is basic')
    feature_names.append(str(i) + 'is acidic')
    feature_names.append(str(i) + 'is exposed (netsurfp)')
    feature_names.append(str(i) + 'RSA score')
    feature_names.append(str(i) + 'ASA score')
    feature_names.append(str(i) + 'p(a-helix)')
    feature_names.append(str(i) + 'p(b-strand)')
    feature_names.append(str(i) + 'p(coil)')
    feature_names.append(str(i) + 'is H-bond donor')
    feature_names.append(str(i) + 'is H-bond acceptor')
    feature_names.append(str(i) + 'is aromatic')
feature_names2 = []
for i in range(-5, 6):
    if i == 0:
        continue
    for j in aanames:
        feature_names2.append(str(i) + j)
    feature_names2.append(str(i) + 'hydrophobicity')
    feature_names2.append(str(i) + 'is hydrophobic')
    feature_names2.append(str(i) + 'flexibility')
    feature_names2.append(str(i) + 'hydrophobicity')
    feature_names2.append(str(i) + 'is hydrophilic')
    feature_names2.append(str(i) + 'surface accessibility')
    feature_names2.append(str(i) + 'surface accessibility')
    feature_names2.append(str(i) + 'pKa')
    feature_names2.append(str(i) + 'is basic')
    feature_names2.append(str(i) + 'is acidic')
    feature_names2.append(str(i) + 'is exposed (netsurfp)')
    feature_names2.append(str(i) + 'RSA score')
    feature_names2.append(str(i) + 'ASA score')
    feature_names2.append(str(i) + 'p(a-helix)')
    feature_names2.append(str(i) + 'p(b-strand)')
    feature_names2.append(str(i) + 'p(coil)')
    feature_names2.append(str(i) + 'is H-bond donor')
    feature_names2.append(str(i) + 'is H-bond acceptor')
    feature_names2.append(str(i) + 'is aromatic')

# print(feature_names[195:234])
# print(len(feature_names))
# print(feature_names)

"""Generate X2, an array identical to the full matrix, but with information on site 0 removed.
Site 0 for a true O-GlcNAc site will always be S/T, but is not in the control set."""
X = fullmatrix
X.shape  # (1928, 429)
X2 = np.delete(X, np.s_[195:234], axis=1)
X2.shape  # (1928, 390)



"""Feature selection"""

X = fullmatrix
# X = X2
y = np.asarray(y)
# X.shape

"""F_test"""
"""I use this feature selection method to tell me what the most informative features are"""


X_indices = np.arange(X.shape[-1])
selector = SelectPercentile(f_classif, percentile=5)  # change percentile to return different numbers of features
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
feature_indices = selector.get_support(indices=True)
# print(selector.coef_)
plt.clf()
plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
        edgecolor='black')
plt.show()

for index in feature_indices:
    print(feature_names[index])  # looks up and prints the name of the top features



"""TSNE clustering"""


X = fullmatrix
y = np.asarray(y)

print(X.shape)
print(y.shape)

X_new = TSNE(n_components=2, perplexity=150).fit_transform(X)
print(X_new.shape)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X_new)
y_new = kmeans.labels_
print(y_new.shape)

score = adjusted_rand_score(y, y_new)

df = pd.DataFrame({'tSNE 1': X_new[:, 0], 'tSNE 2': X_new[:, 1], 'actual_label': y, 'cluster': y_new})
sns.lmplot(data=df, x='tSNE 1', y='tSNE 2', hue='actual_label', fit_reg=False)
sns.lmplot(data=df, x='tSNE 1', y='tSNE 2', hue='cluster', fit_reg=False)
plt.show()


"PCA clustering"


X = fullmatrix
y = np.asarray(y)

n_components = 2
ipca = IncrementalPCA(n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

colors = ['turquoise', 'darkorange']

for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1], ["O-GlcNAc site", "random site"]):
        plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                    color=color, lw=2, label=target_name)

    if "Incremental" in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        plt.title(title + " of O-GlcNAc site flanking features\nMean absolute unsigned error "
                          "%.6f" % err)
    else:
        plt.title(title + " of O-GlcNAc site flanking features")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    # plt.axis([-4, 4, -1.5, 1.5])

plt.show()



"""Classification with SVC"""




X = fullmatrix
y = np.asarray(y)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7, stratify=y)

clf = svm.SVC()
clf.fit(xtrain, ytrain)

# predict the labels for test set using .predict()
prediction = clf.predict(xtest)
# compare predicted to actual labels and calculate the score
score = clf.score(xtest, ytest)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=100)
scores = cross_val_score(clf, X, y, n_jobs=-1, scoring='f1_macro', cv=rskf)

print(score)
print(np.mean(scores))

plt.hist(scores)
plt.show()



"""Random Forest Classifier"""







# X = fullmatrix
X = X2
y = np.asarray(y)

# split data into training and 30% testing (or 70% training)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(xtrain, ytrain)

# predict the labels for test set using .predict()
prediction = clf.predict(xtest)
# compare predicted to actual labels and calculate the score
score = clf.score(xtest, ytest)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=100)
scores = cross_val_score(clf, X, y, n_jobs=-1, scoring='f1_macro', cv=rskf)

print(score)
print(np.mean(scores))

plt.hist(scores)
plt.show()



