#!/bin/bash
USER=$1
PASS=$2
PKG=$3
cd resources
wget --keep-session-cookies --save-cookies=cookies.txt --post-data "username=$USER&password=$PASS&submit=Login" https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition "https://www.cityscapes-dataset.com/file-handling/?packageID=$PKG"
rm -rf *.txt *.html
cd -


