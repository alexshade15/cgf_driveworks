#!/bin/sh

/usr/local/driveworks-5.14/tools/schema/validate_cgfdescriptors.py \
    /usr/local/driveworks-5.14/samples/src/sample_for_forum/cgf_driveworks/PerceptionPipeV0_10/applications/PipelineV0_3.app.json

echo -n "\n\nEND validate_cgfdescriptors\n================================================"
echo -n "\n================================================\n\n"
read var_name

/usr/local/driveworks-5.14/tools/descriptionScheduleYamlGenerator/descriptionScheduleYamlGenerator.py \
    --app /usr/local/driveworks-5.14/samples/src/sample_for_forum/cgf_driveworks/PerceptionPipeV0_10/applications/PipelineV0_3.app.json \
    --output /usr/local/driveworks-5.14/samples/src/sample_for_forum/cgf_driveworks/PerceptionPipeV0_10//

echo -n "\n\nEND descriptionScheduleYamlGenerator\n================================================"
echo -n "\n================================================\n\n"
read var_name

/usr/local/driveworks-5.14/tools/stmcompiler \
 -i /usr/local/driveworks-5.14/samples/src/sample_for_forum/cgf_driveworks/PerceptionPipeV0_10/PipelineV0_3__standardSchedule.yaml \
 -o /usr/local/driveworks-5.14/samples/src/sample_for_forum/cgf_driveworks/PerceptionPipeV0_10/PipelineV0_3__standardSchedule.stm

echo -n "\n\nEND stmcompiler\n================================================"
echo -n "\n================================================\n\n"

/usr/local/driveworks-5.14/tools/stmvizschedule \
-i /usr/local/driveworks-5.14/samples/src/sample_for_forum/cgf_driveworks/PerceptionPipeV0_10/PipelineV0_3__standardSchedule.stm \
-o /usr/local/driveworks-5.14/samples/src/sample_for_forum/cgf_driveworks/PerceptionPipeV0_10/PipelineV0_3__standardSchedule.html

read var_name


scp -r \
/usr/local/driveworks-5.14/samples/src/sample_for_forum/cgf_driveworks/PerceptionPipeV0_10 \
    kineton@10.10.40.11:/home/kineton/nvidia/PerceptionPipeV0_10_nvidia_forum

scp \
/home/perception/build_aarch64/src/sample_for_forum/cgf_driveworks/my_minipipeline/libminipipeline_nodes.so \
    kineton@10.10.40.11:~