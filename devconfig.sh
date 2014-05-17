#! /bin/sh
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.
#
# I'm kind of annoyed by the setuptools, so I have a quick system to smooth
# out some of the bits that bug me. This script is basically a "configure"
# script, but given a scarier name to not confuse people who are just going to
# want to use the standard setuptools.

install_args=""

usage ()
{
    echo "usage: ./devconfig.sh --prefix={DIR}"
}

while [ $# -gt 0 ] ; do
    case "$1" in
	--prefix=*) install_args="$install_args $1" ;;
	*) echo >&2 "error: unrecognized argument \"$1\"" ; usage ; exit 1 ;;
    esac
    shift
done

cat <<EOF >config.status
#! /bin/sh
sed -e "s|@install_args@|$install_args|" <Makefile.in >Makefile
EOF

chmod +x config.status
./config.status
