redo-ifchange $1.pyx

cython -a $1.pyx -o $3
