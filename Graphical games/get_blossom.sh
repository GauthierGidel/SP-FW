mkdir tmp
cd tmp
wget http://pub.ist.ac.at/~vnk/software/blossom5-v2.05.src.tar.gz
tar -xzvf blossom5-v2.05.src.tar.gz
cd blossom5-v2.05.src
make
cp blossom5 ../../
cd ../../
rm -r tmp
