PY_INCL=`python-config --includes`
NP_INCL=`cat numpy_includes`
exec > $3
cat <<-EOF
	OUT="\$1"
	shift
	gcc -bundle -undefined dynamic_lookup ${PY_INCL} ${NP_INCL} -o "\$OUT" "\$@"
EOF
chmod a+x $3
