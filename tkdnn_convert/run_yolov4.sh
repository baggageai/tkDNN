main_path="/local/cigrate_cigar/"
weights=$(find -L $main_path -name "*.weights")
cfg=$(find -L $main_path -name "*.cfg")
names=$(find -L $main_path -name "*.names")
echo "data: $weights $cfg $names"
#Removing build folder of the project directory
#Removing build folder of home directory to overcome overwriting issue
if [ -d ~/"tkDNN/" ]; then
  rm -rf ~/tkDNN/
fi

#Removing build folder of the project directory
if [ -d "tkDNN/" ]; then
  rm -rf tkDNN/
fi
git config --global http.sslverify false
git clone https://github.com/ceccocats/tkDNN.git
cd tkDNN
git clone https://git.hipert.unimore.it/fgatti/darknet.git
cd darknet
make -j16
mkdir layers debug
./darknet export $cfg $weights layers
cd ..
mkdir build
mkdir build/yolo4
cp -r darknet/layers build/yolo4/
cp -r darknet/debug build/yolo4/

sed -i "s#../tests/darknet/cfg/yolo4.cfg#$cfg#" tests/darknet/yolo4.cpp
sed -i "s#../tests/darknet/names/coco.names#$names#" tests/darknet/yolo4.cpp

cd build
cmake ..
make -j16
./test_yolo4
mv *.rt $main_path
echo "completed"

