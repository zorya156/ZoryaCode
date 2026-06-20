import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

fig,ax = plt.subplots()

x = np.random.rand(1488)
y = np.random.rand(1488)
ax.plot(np.square(x), np.square(y))
plt.show()





# data = [[1,4,8,8], [6,7,6,9], [5,2,4,2]]
# df = pd.DataFrame(data,  index = ['Z','O','V'],columns=['S', 'V', 'O', 'GOYDA'])
# print(df)