CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
echo "CUDA version: $CUDA_VER"

CUDA_VER=$CUDA_VER make -C nvdsinfer_custom_impl_Yolo clean
CUDA_VER=$CUDA_VER make -C nvdsinfer_custom_impl_Yolo

CUDA_VER=$CUDA_VER make -C nvdsinfer_custom_impl_Yolo_face clean
CUDA_VER=$CUDA_VER make -C nvdsinfer_custom_impl_Yolo_face

if [ ! -d logs/ ]
then
    mkdir logs/
fi

python3 deepstream_pipeline.py
