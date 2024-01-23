Where to put PerceptionPipeV0_10?
    You can put it wherever you prefer.

Where to put my_minipipeline?
    Please, put my_minipipeline under /usr/local/driveworks-5.14/samples/src/

Unfortunately at the moment YOLO path is hard coded. Please update it.
You can find it in ./cgf_driveworks/my_minipipeline/detectAndTrackImpl.cpp, line 258.
yolo.bin.json and yolo.bin are currently stored under ./cgf_driveworks/

How to enable PERCEPTION_ENABLED?
    Please, add to /usr/local/driveworks-5.14/samples/CMakeLists.txt
    the following line: add_compile_definitions(PERCEPTION_ENABLED=1)

How to build my_minipipeline?
    If you saved my_minipipeline under /usr/local/driveworks-5.14/samples/src/ as I requested,
    please, add to /usr/local/driveworks-5.14/samples/CMakeLists.txt
    the following line: add_subdirectory(src/minipipeline_merge)

  Then:
    cmake \
    -B <ANY_BUILD_FOLDER_YOU_PREFER> \
    -DCMAKE_TOOLCHAIN_FILE=/usr/local/driveworks/samples/cmake/Toolchain-V5L.cmake \
    -DVIBRANTE_PDK=/drive/drive-linux \
    -S /usr/local/driveworks-5.14/samples

How to run?
    cd ./my_minipipeline/PerceptionPipeV0_10/bashscript/
    sudo ./run_mini_pipeline3.sh 

Where is the instruction that gives me problem?
    ./cgf_driveworks/my_minipipeline/detectAndTrackImpl.cpp, line 658.

How to disable object tracking, leaving only object detection running?
    In ./cgf_driveworks/my_minipipeline/detectAndTrackImpl.cpp
    Comment line 177
    Uncomment lines 531 and 562-567

Troubleshooting
    -Please, make sure that the .so file called is mine .so and not yours.

    - ./cgf_driveworks/PerceptionPipeV0_10/dataset folder is a copy of /usr/local/driveworks-5.14/data/samples/minipipeline/dataset
      I will remove irisSimShort_2217.bin since it is too big to be uploaded, but you can find it under /usr/local/driveworks-5.14/data/samples/minipipeline/dataset



How did I generated stub, yaml and stm?
    ./cgf_driveworks/PerceptionPipeV0_10/TUTORIAL.txt


Thank you for your help. I really appreciate it.