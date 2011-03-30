redo-ifchange $1.o
redo-ifchange t/pyLD

rm -f $1.so
t/pyLD $3 $1.o
