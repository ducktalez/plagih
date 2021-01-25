import sympy
import matplotlib.pyplot as plt

ps = list(sympy.primerange(0, 100000))
psum = 0
psum_list = [0]
fuk = 1
for p in ps:
       psum+= p * fuk
       fuk *= -1
       psum_list.append(psum)
print(psum_list)
#plt.plot([abs(x) for x in psum_list])
#plt.show()
print(psum_list[1::2])
plt.plot(psum_list[::2])
