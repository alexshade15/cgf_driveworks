#!/bin/sh


echo "======================================================================="
echo "libs.so:"

if ls /home/kineton/lib*.so ; then
    sudo mv /home/kineton/lib*.so /usr/local/driveworks-5.14/bin/
    echo "libs.so copied from /home/kineton to dw/bin"
else
    echo "libs not found."
fi

echo " "
echo " "
echo " ----- "


if cd ../logSpace/LogFolder ; then
    rm -rf *
    echo "CEANED ${PWD}"
else 
    echo "folder not found. EXIT."
    exit
fi

echo " "
echo " ----- "

cd ../../bashscript/
echo "back to previus directory, run pipeline"

echo " "
echo " "
echo " ------------------------- "
echo " ------------------------- "


sudo ./run_mini_pipeline3.sh 