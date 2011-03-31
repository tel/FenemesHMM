PY_INCL=`python-config --includes`
NP_INCL=`cat numpy_includes`

if [ `uname` == 'Linux' ]
then
	FLAGS="-shared"
else
	FLAGS="-bundle -undefined dynamic_lookup"
fi

exec > $3
cat <<-EOF
	OUT="\$1"
	shift
	gcc $FLAGS ${PY_INCL} ${NP_INCL} -o "\$OUT" "\$@"
EOF
chmod a+x $3
