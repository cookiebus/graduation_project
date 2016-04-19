sudo pt-get install python-scipy
sudo apt-get install libpng-dev
cd ~/Downloads
wget http://download.savannah.gnu.org/releases/freetype/freetype-2.4.10.tar.gz
tar zxvf freetype-2.4.10.tar.gz
cd freetype-2.4.10/
./congfigure
make
sudo make install
