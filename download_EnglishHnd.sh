#!/bin/sh

wget http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz
tar -xvf EnglishHnd.tgz English/Hnd/Img/Sample001/
mkdir CUSTOM_DATASET
find English/ -iname '*.png' -exec mv '{}' CUSTOM_DATASET/ \;
rm -rf English
