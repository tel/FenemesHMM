import numpy as np
import main
import jumps
import Image

foo = lambda: jumps.jump_matrices(main.d.baseforms[0], main.fenomes)

a, b, c, d, e = foo()
Image.fromarray(np.uint8(255*e/np.max(e))).save('out.png')
