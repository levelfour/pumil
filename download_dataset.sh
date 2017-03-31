#!/bin/sh

usage() {
    cat 1>&2 << EOF
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
OVERVIEW: download and uncompress datasets from UCI repository.
  
USAGE:
      $(basename ${0}) [dataset]
  
DATASET:
 musk1          drug activity (version 1)
 musk2          drug activity (version 2)
 elephant       Corel Image Set (Elephant)
 fox            Corel Image Set (Fox)
 tiger          Corel Image Set (Tiger)
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
EOF
exit -1
}

dl_trec9() {
    if [ ! -d MilData ]; then
      curl -s -O http://www.cs.columbia.edu/~andrews/mil/data/MIL-Data-2002-Musk-Corel-Trec9.tgz
      tar xf MIL-Data-2002-Musk-Corel-Trec9.tgz
    fi

    if [ "$1" = "elephant" ]; then
      mv MilData/Elephant/data_100x100.svm elephant.data
    elif [ "$1" = "fox" ]; then
      mv MilData/Fox/data_100x100.svm      fox.data
    elif [ "$1" = "tiger" ]; then
      mv MilData/Tiger/data_100x100.svm    tiger.data
    elif [ "$1" = "musk1" ]; then
      mv MilData/Musk/musk1norm.svm        musk1.data
    elif [ "$1" = "musk2" ]; then
      mv MilData/Musk/musk2norm.svm        musk2.data
    fi
}

if [ "${1}" = "" ]; then
    usage
fi

case ${1} in
musk1)
    dl_trec9 musk1;;
musk2)
    dl_trec9 musk2;;
elephant)
    dl_trec9 elephant;;
fox)
    dl_trec9 fox;;
tiger)
    dl_trec9 tiger;;
--help | -h)
    usage;;
*)
    echo "error: Unknown dataset name '${1}'" 1>&2
    usage;;
esac
