redo-ifchange $1.pyx

cython -a $1.pyx -o $3
mv ${3%.tmp}.html $1.html
