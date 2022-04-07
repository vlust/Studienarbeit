
import matplotlib.pyplot as plt
import os

my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
my_file = '/graph.png'
 
print(my_path)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(100))

fig.savefig(my_path+ my_file)