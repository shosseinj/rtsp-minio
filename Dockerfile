
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

RUN pip install flask --ignore-installed blinker
