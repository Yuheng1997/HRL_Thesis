import os
import numpy as np


if __name__ == '__main__':
    path = os.path.join(os.path.abspath(os.getcwd()), "datasets/train/data.tsv")
    print(path)
    
    data = np.loadtxt(path, delimiter='\t').astype(np.float32)

    print(len(data))

    data_10 = np.concatenate((
        data[:1800], data[18000:18900], data[27000:27900],
        data[36000:36200], data[38000:38100], data[39000:39100],
    ))

    data_1 = np.concatenate((
        data[:180], data[18000:18090], data[27000:27090],
        data[36000:36200], data[38000:38100], data[39000:39100],
    ))

    np.savetxt("data_10%.tsv", data_10, delimiter='\t', fmt="%.10f")
    np.savetxt("data_1%.tsv", data_1, delimiter='\t', fmt="%.10f")
