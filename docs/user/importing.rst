Importing Data
==============

Hierarch is compatible with pandas DataFrames and numpy arrays. 
Pandas is capable of conveniently importing data from a wide variety 
of formats, including Excel files. ::

    import pandas as pd
    data = pd.read_excel(filepath)