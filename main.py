import numpy as np
import matplotlib.pyplot as plt

#1

z = np.arange(0, 11, 2)
k = np.arange(30, 51, 4)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('K as a function of Z')
ax1.plot(z, k)
ax2.stem(z, k)

for ax in (ax1, ax2):
    ax.set(xlabel='z', ylabel='k')
plt.savefig("1.png")
plt.show()


#2
matrixA = np.array([1, 2, 3, 4, 9, 8, 3, 5, 6]).reshape(3, 3)
matrixB = np.array([5, 8, 23, 10, 7, 1, 3, 2, 1]).reshape(3, 3)
matrixC = matrixA + matrixB
matrixD = matrixA * matrixB
matrixE = np.transpose(matrixA) + np.transpose(matrixB)


#3



# plt.subplot(1, 2, 1)
# plt.stem(z, k)
# plt.title("K as a function of Z")
# plt.xlabel("z")
# plt.ylabel("k")
# plt.savefig("1.2.png")
# plt.show()

# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()
#
# X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
# C,S = np.cos(X), np.sin(X)
#
# plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
# plt.plot(X, S, color="red", linewidth=2.5, linestyle="-")
# plt.savefig('plot.png')
#
#
# plt.xlim(X.min()*1.1, X.max()*1.1)
# plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
# [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
#
# plt.ylim(C.min()*1.1,C.max()*1.1)
# plt.yticks([-1, 0, +1],
# [r'$-1$', r'$0$', r'$+1$'])
#
# plt.show()
