#!/usr/bin/env python
# import numpy as np
# import matplotlib.pyplot as plt
# import Image
# from PIL import ImageFilter
# import cv2


import numpy as np
import matplotlib.pyplot as plt
import Image
from PIL import ImageFilter

I = Image.open('/home/wpoh/myStuff/aranz_silhouette/python_proj/img/wound10.jpg')
I = I.filter(ImageFilter.BLUR)
I=I.convert('L')
p = np.asarray(I).astype('int8')
w,h = I.size
x, y = np.mgrid[0:h:500j, 0:w:500j]

dy, dx = np.gradient(p)
skip = (slice(None, None, 3), slice(None, None, 3))

fig, ax = plt.subplots()
im = ax.imshow(I.transpose(Image.FLIP_TOP_BOTTOM),
               extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar(im)
ax.quiver(x[skip], y[skip], dx[skip].T, dy[skip].T)

ax.set(aspect=1, title='Quiver Plot')
plt.show()