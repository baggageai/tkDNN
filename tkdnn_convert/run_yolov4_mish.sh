main_path="/local/lighters/"
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
mkdir build/yolo4x
cp -r darknet/layers build/yolo4x/
cp -r darknet/debug build/yolo4x/
sed -i "s#std::string(TKDNN_PATH) +##" tests/darknet/yolo4x.cpp
sed -i "s#/tests/darknet/cfg/yolo4x.cfg#$cfg#" tests/darknet/yolo4x.cpp
sed -i "s#/tests/darknet/names/coco.names#$names#" tests/darknet/yolo4x.cpp

cd build
cmake ..
make -j16
./test_yolo4x
mv *.rt $main_path
echo "completed"



