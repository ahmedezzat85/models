@echo off

set MODEL_DIR="D:\Model_Zoo\Tensorflow\ssd_mobilenet_v1_coco_2017_11_17"
set CONFIG_FILE="D:\GitHub\object_detection\samples\configs\ssd_mobilenet_v1_coco.config"
set INPUT_TENSOR=input
set OUT_DIR=ssd_mobile_net

python export_inference_graph.py ^
    --input_type %INPUT_TENSOR% ^
    --pipeline_config_path %CONFIG_FILE% ^
    --trained_checkpoint_prefix %MODEL_DIR%\model.ckpt ^
    --output_directory %OUT_DIR% ^
    --input_shape 1,300,300,3
