import glob 
import os
import csv

d = {'a':[1,2],'b':[3,4]}
with open('testwrite.csv','wb') as outfile:
    fieldnames = d.keys()
    writer = csv.writer(outfile)
    writer.writerow(d.keys())

    # Assumes each key has same number of elements
    
    sample = d['a']
    n_rows = len(sample)
    for i in range(n_rows):
        a,b = d['a'][i],d['b'][i]
        writer.writerow([a,b])
    

