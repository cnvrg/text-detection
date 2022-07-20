apt-get install -y libmagic1
pip install --upgrade pip
apt-get -y update
apt-get install -y git
pip install git+https://github.com/jaidedai/easyocr.git
apt-get install -y libgl1-mesa-dev
apt-get install -y libfreetype6-dev
pip uninstall -y pillow
pip install --no-cache-dir pillow