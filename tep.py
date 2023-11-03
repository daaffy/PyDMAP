import betterspy
from scipy import sparse

A = sparse.rand(44991, 44991, density=0.001)

# betterspy.plot()
# set attributes on gca()
# plt.show()
# or directly

betterspy.show(A)