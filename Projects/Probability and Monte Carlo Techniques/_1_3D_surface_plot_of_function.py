# 3D Surface plot of a funciton of two varibles
# coding: utf-8

# In[95]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(7., 5.))
ax = Axes3D(fig)
costh = np.linspace(-1, 1, endpoint=True) #the default size of the sample is 50
pmu = np.linspace(-1, 1, endpoint=True) 
costh, pmu = np.meshgrid(costh, pmu)
f = 1./2.*(1.-1./3.*pmu*costh)

ax.plot_surface(costh, pmu, f, rstride=1, cstride=1, cmap=plt.cm.hot) 
ax.contourf(costh, pmu, f, zdir='z', offset=-2, cmap=plt.cm.hot)
ax.set_xlabel(r'$cos \ \theta $', labelpad=10) #lavelpad positionate the axis label
ax.set_ylabel(r'$P_\mu$', labelpad= 10)
ax.set_zlabel(r'$f$')

plt.savefig(r"C:\Users\Aleix López\Desktop\3d_pdf.jpg")

plt.show() #You should call savefig and savetxt before calling show.


# In[ ]:




