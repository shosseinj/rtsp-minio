We currently provide the following sample applications:

- [deepstream-test1](apps/deepstream-test1) -- 4-class object detection pipeline, also demonstrates support for new nvstreammux
- [deepstream-test2](apps/deepstream-test2) -- 4-class object detection, tracking and attribute classification pipeline
- [deepstream-test3](apps/deepstream-test3) -- multi-stream pipeline performing 4-class object detection, also supports triton inference server, no-display mode, file-loop and silent mode
- [deepstream-test4](apps/deepstream-test4) -- msgbroker for sending analytics results to the cloud
- [deepstream-imagedata-multistream](apps/deepstream-imagedata-multistream) -- multi-stream pipeline with access to image buffers
- [deepstream-test1-usbcam](apps/deepstream-test1-usbcam) -- deepstream-test1 pipeline with USB camera input
- [deepstream-test1-rtsp-out](apps/deepstream-test1-rtsp-out) -- deepstream-test1 pipeline with RTSP output, demonstrates adding software encoder option to support Jetson Orin Nano
- [deepstream-opticalflow](apps/deepstream-opticalflow) -- optical flow and visualization pipeline with flow vectors returned in NumPy array
- [deepstream-segmentation](apps/deepstream-segmentation) -- segmentation and visualization pipeline with segmentation mask returned in NumPy array
- [deepstream-nvdsanalytics](apps/deepstream-nvdsanalytics) -- multistream pipeline with analytics plugin
- [runtime_source_add_delete](apps/runtime_source_add_delete) -- add/delete source streams at runtime
- [deepstream-imagedata-multistream-redaction](apps/deepstream-imagedata-multistream-redaction) -- multi-stream pipeline with face detection and redaction **NOTE** now uses local sink instead of rtsp output
- [deepstream-rtsp-in-rtsp-out](apps/deepstream-rtsp-in-rtsp-out) -- multi-stream pipeline with RTSP input/output - has command line option "--rtsp-ts" for configuring the RTSP source to attach the timestamp rather than the streammux
- [deepstream-preprocess-test](apps/deepstream-preprocess-test) -- multi-stream pipeline using nvdspreprocess plugin with custom ROIs
- [deepstream-demux-multi-in-multi-out](apps/deepstream-demux-multi-in-multi-out) -- multi-stream pipeline using nvstreamdemux plugin to generated separate buffer outputs
- [deepstream-imagedata-multistream-cupy](apps/deepstream-imagedata-multistream-cupy) -- access imagedata buffer from GPU in a multistream source as CuPy array - x86 only
- [deepstream-segmask](apps/deepstream-segmask) -- access and interpret segmentation mask information from NvOSD_MaskParams
- [deepstream-custom-binding-test](apps/deepstream-custom-binding-test) -- demonstrate usage of NvDsUserMeta for attaching custom data structure - see also the [Custom User Meta Guide](bindings/CUSTOMUSERMETAGUIDE.md)

Detailed application information is provided in each application's subdirectory under [apps](apps).

docker run -it --rm --gpus all --ipc host -p 5000:5000 -p 8554:8554 -p 7070:7070 -v "${PWD}:/workspace" deepstream:v5

python3 deepstream_imagedata-multistream.py file:///workspace/video4.mp4 frames1
