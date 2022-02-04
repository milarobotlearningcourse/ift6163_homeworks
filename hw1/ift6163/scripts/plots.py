from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col


x = np.arange(0, 5, 1)

avg_return_ant =np.asarray([ 4667.8720703125, 4504.46044921875, 4746.77294921875, 4608.52734375, 4677.94140625])
std_return_ant =np.asarray([60.305023193359375, 711.9349365234375, 138.71470642089844, 50.87261199951172, 113.11861419677734])

avg_return_humanoid =np.asarray([329.11407470703125,  380.49615478515625, 634.4009399414062, 915.8148193359375, 1158.507080078125])
std_return_humanoid =np.asarray([78.93054962158203, 69.61216735839844, 196.55804443359375, 367.2467041015625, 446.755126953125])

fig, ax = plt.subplots()

# plt.plot(x, avg_return_ant)
ax.errorbar(x, avg_return_ant,
            yerr=std_return_ant,
            fmt='-o', label="Ant-DAgger", color="b")

plt.hlines(4745.2470703125, 0, 4, color="b", label="Ant-expert")

ax.errorbar(x, avg_return_humanoid,
            yerr=std_return_humanoid,
            fmt='-o', label="Humanoid-DAgger", color="r")
plt.hlines(10690.314453125, 0, 4, color="r", label="Humanoid-expert")

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_title('Line plot with error bars')

ax.legend()

plt.savefig('q2.jpg')