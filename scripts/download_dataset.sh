#!/usr/bin/env bash

set -e

DATA_DIR="LJSpeech-1.1"

LJS_ARCH="LJSpeech-1.1.tar.bz2"
LJS_URL="http://data.keithito.com/data/speech/${LJS_ARCH}"

if [ ! -f ${LJS_ARCH} ]; then
  echo "Downloading ${LJS_ARCH} ..."
  wget -q ${LJS_URL}
fi

