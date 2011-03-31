redo-ifchange numpy_includes
PY_INCL=`python-config --includes`
NP_INCL=`cat numpy_includes`
exec >$3
cat <<-EOF
	gcc -fno-strict-aliasing -g -fPIC -O2 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes \
	${NP_INCL} \
	${PY_INCL} \
	-o /dev/fd/1 \
	-c "\$1"
EOF
chmod a+x $3
