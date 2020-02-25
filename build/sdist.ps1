# Windows analogue of sdist.sh.

rm *.tar.gz
build_dir=$PWD
pushd ../.. # need to run in same directory as setup.py
# assumes container has Python 3.5 installed
$Env:Path += ";C:\Users\ContainerAdministrator\AppData\Local\Programs\Python\Python35;C:\Users\ContainerAdministrator\AppData\Local\Programs\Python\Python35\Scripts"
echo "Python version installed (>= 3.5 required):" 
python.exe --version
   pip install cython # not present in GitHub macos-10.15 environment
   python.exe setup.py sdist --dist-dir=$build_dir
rm -rf esig.egg-info
popd
