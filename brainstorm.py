# This is a random file which is only there for random code to test.

import matplotlib.pyplot as plt

xvals = [i for i in range(0, 10)]
yvals1 = [i**2 for i in range(0, 10)]
yvals2 = [i**3 for i in range(0, 10)]

f, plt = plt.subplots(1)
plt.plot(xvals, yvals1)
plt.plot(xvals, yvals2)
plt.yscale('linear')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
# plt.savefig('rewards.jpg')
plt.show()
plt.close()
