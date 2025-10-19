import gi

gi.require_version("Gst", "1.0")

from gi.repository import GLib, Gst
import pyds

import os, sys, time
from threading import Lock
from ctypes import sizeof, c_float
import ctypes

import cupy as cp
from cupyx.scipy import ndimage

# sys.path.append("/opt/nvidia/deepstream/deepstream/lib")

import logging

ACTIVITY = "DEEPSTREAM PIPELINE"

logfile = ACTIVITY.strip().replace(" ", "_")

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger = logging.getLogger(logfile)

from logging.handlers import RotatingFileHandler

Rhandler = RotatingFileHandler(
    f"logs/{logfile}.log", maxBytes=20 * 1024 * 1024, backupCount=20
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
Rhandler.setFormatter(formatter)
logger.addHandler(Rhandler)
logger.setLevel(LOGLEVEL)
logger.info(f"Running {ACTIVITY}")

MAX_ELEMENTS_IN_DISPLAY_META = 16

SOURCE = ""
INFER_CONFIG = ""
STREAMMUX_BATCH_SIZE = 1
STREAMMUX_WIDTH = 1280
STREAMMUX_HEIGHT = 720
GPU_ID = 0
BATCH_SIZE = 1

PERF_MEASUREMENT_INTERVAL_SEC = 5
JETSON = False

INPUT_FILE_NAME = "test_cctv.mp4"
OUTPUT_FILE_NAME = "output.mp4"
YOLOV8_FACE_CONFIG_FILE = "yolov8_face_config.txt"

TRACKER_WIDTH = 640
TRACKER_HEIGHT = 640
TRACKER_CONFIG_FILE_PATH = os.path.abspath("config_tracker_NvDCF_accuracy.yml")

perf_struct = {}

class GETFPS:
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.start_time = time.time()
        self.is_first = True
        self.frame_count = 0
        self.total_fps_time = 0
        self.total_frame_count = 0
        self.fps_lock = Lock()

    def update_fps(self):
        with self.fps_lock:
            if self.is_first:
                self.start_time = time.time()
                self.is_first = False
                self.frame_count = 0
                self.total_fps_time = 0
                self.total_frame_count = 0
            else:
                self.frame_count = self.frame_count + 1

    def get_fps(self):
        with self.fps_lock:
            end_time = time.time()
            current_time = end_time - self.start_time
            self.total_fps_time = self.total_fps_time + current_time
            self.total_frame_count = self.total_frame_count + self.frame_count
            current_fps = float(self.frame_count) / current_time
            avg_fps = float(self.total_frame_count) / self.total_fps_time
            self.start_time = end_time
            self.frame_count = 0
        return current_fps, avg_fps

    def perf_print_callback(self):
        if not self.is_first:
            current_fps, avg_fps = self.get_fps()
            logger.info(f"Stream {self.stream_id + 1} - FPS: {current_fps:.2f} ({avg_fps:.2f})\n")
        return True

def blur_face(frame_meta, gst_buffer):
       # Create dummy owner object to keep memory for the image array alive
        owner = None
        # Getting Image data using nvbufsurface
        # the input should be address of buffer and batch_id
        # Retrieve dtype, shape of the array, strides, pointer to the GPU buffer, and size of the allocated memory
        data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(hash(gst_buffer), frame_meta.batch_id)
        # dataptr is of type PyCapsule -> Use ctypes to retrieve the pointer as an int to pass into cupy
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        # Get pointer to buffer and create UnownedMemory object from the gpu buffer
        c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
        unownedmem = cp.cuda.UnownedMemory(c_data_ptr, size, owner)
        # Create MemoryPointer object from unownedmem, at index 0
        memptr = cp.cuda.MemoryPointer(unownedmem, 0)
        # Create cupy array to access the image data. This array is in GPU buffer
        n_frame_gpu = cp.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C')

        # # Get frame dimensions
        frame_h, frame_w = n_frame_gpu.shape[0], n_frame_gpu.shape[1]

        # Initialize cuda.stream object for stream synchronization
        stream = cp.cuda.stream.Stream(null=True)
        sigma_val = max(1, int(min(frame_w, frame_h) * 0.005))
        # Modify the red channel to add blue tint to image
        with stream:
            # Iterate over each object detected in the frame
            l_obj = frame_meta.obj_meta_list
            to_remove = []
            while l_obj is not None:
                try:
                    # Cast l_obj.data to NvDsObjectMeta
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break # No more objects

                # Get bounding box coordinates
                rect_params = obj_meta.rect_params

                # Clip coordinates to be within frame boundaries (prevents errors)
                top = max(0, int(rect_params.top))
                left = max(0, int(rect_params.left))
                bottom = min(frame_h, int(rect_params.top + rect_params.height))
                right = min(frame_w, int(rect_params.left + rect_params.width))
                
                # Calculate width and height *after* clipping
                height = bottom - top
                width = right - left

                # Only process if the ROI is valid (has width and height)
                if width > 0 and height > 0:
                    # Extract the Region of Interest (ROI) from the frame
                    face_roi = n_frame_gpu[top:bottom, left:right]
                    
                    # Ensure sigma is at least 1 to apply some blur
                    if sigma_val > 0:
                        # Apply Gaussian blur directly to the GPU buffer ROI (in-place)
                        # We blur in Y (dim 0) and X (dim 1), but not across channels (dim 2)
                        ndimage.gaussian_filter(
                            face_roi, 
                            sigma=(sigma_val, sigma_val, 0), # (sigma_y, sigma_x, sigma_channel)
                            output=face_roi  # Perform the operation in-place
                        )
            
                try:
                    # Move to the next object
                    l_obj = l_obj.next
                    to_remove.append(obj_meta)
                except StopIteration:
                    break
            # Remove objects that were processed
            for obj in to_remove:
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj)
        stream.synchronize()

def nvosd_sink_pad_buffer_probe(pad, info, user_data):
    buf = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        blur_face(frame_meta, buf)
        perf_struct[frame_meta.source_id].update_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def cb_newpad(decodebin, decoder_src_pad, data):
    try:
        logger.info("In cb_newpad")
        caps = decoder_src_pad.get_current_caps()
        if not caps:
            caps = decoder_src_pad.query_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)

        # Need to check if the pad created by the decodebin is for video and not
        # audio.
        logger.info(f"gstname ==> {gstname}")
        if gstname.find("video") != -1:
            # Link the decodebin pad only if decodebin has picked nvidia
            # decoder plugin nvdec_*. We do this by checking if the pad caps contain
            # NVMM memory features.
            logger.info(f"Linking decodebin pad to source bin ghost pad")
            if features.contains("memory:NVMM"):
                # Get the source bin ghost pad
                bin_ghost_pad = source_bin.get_static_pad("src")
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    logger.error("Failed to link decoder src pad to source bin ghost pad")
            else:
                logger.error("Error: Decodebin did not pick nvidia decoder plugin.")
    except Exception as err:
        logger.error(f"Error in cb_newpad, Error ==> {err}")

def decodebin_child_added(child_proxy, Object, name, user_data):
    try:
        logger.info(f"Decodebin child added: {name}")
        if name.find("decodebin") != -1:
            Object.connect("child-added", decodebin_child_added, user_data)
    except Exception as err:
        logger.error(f"Error in decodebin_child_added, Error ==> {err}")

def create_source_bin(index, uri):
    try:
        logger.info(f"Creating source bin {index} for {uri}")

        # Create a source GstBin to abstract this bin's content from the rest of the
        # pipeline
        bin_name = "source-bin-%02d" % index
        logger.info(f"Creating source bin {bin_name}")
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            logger.error("Unable to create source bin")

        # Source element for reading from the uri.
        # We will use decodebin and let it figure out the container format of the
        # stream and the codec and plug the appropriate demux and decode plugins.
        uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
        if not uri_decode_bin:
            logger.error("Unable to create uri decode bin")

        # We set the input uri to the source element
        uri_decode_bin.set_property("uri", uri)

        # Connect to the "pad-added" signal of the decodebin which generates a
        # callback once a new pad for raw data has beed created by the decodebin
        uri_decode_bin.connect("pad-added", cb_newpad, nbin)
        uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

        # We need to create a ghost pad for the source bin which will act as a proxy
        # for the video decoder src pad. The ghost pad will not have a target right
        # now. Once the decode bin creates the video decoder and generates the
        # cb_newpad callback, we will set the ghost pad target to the video decoder
        # src pad.
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            logger.error("Failed to add ghost pad in source bin")
            return None

        return nbin
    except Exception as err:
        logger.error(f"Error in create source bin, Error ==> {err}")
        return None

def bus_call(bus, message, user_data):
    loop = user_data
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.debug("EOS")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        error, debug = message.parse_warning()
        logger.warning(f"{error.message} - {debug}")
    elif t == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
        logger.error(f"{error.message} - {debug}")
        loop.quit()
    return True

def main():
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    logger.info("Creating pipeline...")
    pipeline = Gst.Pipeline()

    if not pipeline:
        logger.error("Unable to create pipeline")
    
    # Create nvstreammux plugin to form batches from one or more sources
    streammux_name = "storesense_stream-muxer_"
    streammux = Gst.ElementFactory.make("nvstreammux", streammux_name)

    if not streammux:
        logger.error("Unable to create streammux")
    
    # Add streammux to the pipeline
    pipeline.add(streammux)

    
    # Setting streammux properties
    """
        nvstreammux expects to set fixed width and height for the input sources.
        current interpolation-method is 4 i.e Lancsoz interpolation.
    """
    streammux.set_property("width", STREAMMUX_WIDTH)
    streammux.set_property("height", STREAMMUX_HEIGHT)
    streammux.set_property("batch-size", STREAMMUX_BATCH_SIZE)
    streammux.set_property("batched-push-timeout", 4000000)
    streammux.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_DEVICE))
    streammux.set_property("interpolation-method", 4)
    streammux.set_property("enable-padding", 1)
    streammux.set_property("gpu-id", GPU_ID)

    stream_id = 0
    filepath = "file://" + os.path.abspath(INPUT_FILE_NAME)
    source_bin = create_source_bin(stream_id, filepath)
    pipeline.add(source_bin)

    # Link to streammux
    padname = "sink_%u" % stream_id
    sinkpad = streammux.request_pad_simple(padname)
    srcpad = source_bin.get_static_pad("src")
    srcpad.link(sinkpad)

    perf_struct[stream_id] = GETFPS(stream_id)

    # Yolov8 Face
    logger.info("Creating yolov8_face_nvinference")
    yolov8_face_name = "yolov8_face"
    yolov8_face_nvinference = Gst.ElementFactory.make("nvinfer", yolov8_face_name)
    if not yolov8_face_nvinference:
        logger.error("Unable to create yolov8_face_nvinference")
        return -1

    yolov8_face_nvinference.set_property("batch-size", STREAMMUX_BATCH_SIZE)
    yolov8_face_nvinference.set_property("config-file-path", YOLOV8_FACE_CONFIG_FILE)
    yolov8_face_nvinference.set_property("gpu-id", GPU_ID)

    pipeline.add(yolov8_face_nvinference)
    streammux.link(yolov8_face_nvinference)

    # Face Tracker
    logger.info("Creating face_tracker")
    face_tracker_name = "face_tracker"
    face_tracker = Gst.ElementFactory.make("nvtracker", face_tracker_name)
    if not face_tracker:
        logger.error("Unable to create face_tracker")
        return -1
    
    face_tracker.set_property("tracker-width", TRACKER_WIDTH)
    face_tracker.set_property("tracker-height", TRACKER_HEIGHT)
    face_tracker.set_property("display-tracking-id", 1)
    face_tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    face_tracker.set_property("ll-config-file", TRACKER_CONFIG_FILE_PATH)

    pipeline.add(face_tracker)
    yolov8_face_nvinference.link(face_tracker)

    # nvvidconv to convert NVMM to RGBA
    logger.info("Creating nvvidconv")
    nvvidconv_name = "nvvidconv"
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", nvvidconv_name)
    if not nvvidconv:
        logger.error("Unable to create nvvidconv")
        return -1

    nvvidconv.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_DEVICE))
    nvvidconv.set_property("gpu-id", GPU_ID)
    pipeline.add(nvvidconv)
    face_tracker.link(nvvidconv)

    # nvvidconv to convert NVMM to RGBA
    logger.info("Creating capsfilter")
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    if not capsfilter:
        logger.error("Unable to create capsfilter")
        return -1

    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    capsfilter.set_property("caps", caps)
    pipeline.add(capsfilter)
    nvvidconv.link(capsfilter)

    logger.info("Creating nvosd")
    nvosd = Gst.ElementFactory.make("nvdsosd", "nvdsosd")
    if not nvosd:
        logger.error("Unable to create nvosd")
        return -1

    nvosd.set_property("process-mode", int(pyds.MODE_GPU))
    nvosd.set_property("qos", 0)
    nvosd.set_property("gpu-id", GPU_ID)
    pipeline.add(nvosd)
    capsfilter.link(nvosd)

    logger.info("Creating nvvidconv2")
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert2")
    if not nvvidconv2:
        logger.error("Unable to create nvvidconv2")
        return -1

    nvvidconv2.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_DEVICE))
    nvvidconv2.set_property("gpu-id", GPU_ID)
    pipeline.add(nvvidconv2)
    nvosd.link(nvvidconv2)

    logger.info("Creating capsfilter2")
    capsfilter2 = Gst.ElementFactory.make("capsfilter", "capsfilter2")
    if not capsfilter2:
        logger.error("Unable to create capsfilter")
        return -1

    caps2 = Gst.Caps.from_string("video/x-raw(memory:NVMM)")
    capsfilter2.set_property("caps", caps2)
    pipeline.add(capsfilter2)
    nvvidconv2.link(capsfilter2)

    logger.info("Creating H.264 encoder")
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "video-encoder")
    if not encoder:
        logger.error("Unable to create 'nvv4l2h264enc' encoder")
        return -1
    encoder.set_property("bitrate", 4000000) # 4 Mbps
    
    logger.info("Creating H.264 parser")
    parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not parser:
        logger.error("Unable to create 'h264parse' parser")
        return -1
    
    logger.info("Creating MP4 muxer")
    muxer = Gst.ElementFactory.make("qtmux", "mp4-muxer")
    if not muxer:
        logger.error("Unable to create 'qtmux' muxer")
        return -1

    logger.info("Creating File sink")
    filesink = Gst.ElementFactory.make("filesink", "file-sink")
    if not filesink:
        logger.error("Unable to create 'filesink'")
        return -1
    
    # Set the output file path
    filesink.set_property("location", OUTPUT_FILE_NAME)
    # Set sync=0 to save as fast as possible, not in real-time
    filesink.set_property("async", 0)
    filesink.set_property("sync", 0) 

    # Add new elements to the pipeline
    pipeline.add(encoder)
    pipeline.add(parser)
    pipeline.add(muxer)
    pipeline.add(filesink)

    # Link the new elements
    if not capsfilter2.link(encoder):
        logger.error("Failed to link nvosd to encoder")
        return -1
    if not encoder.link(parser):
        logger.error("Failed to link encoder to parser")
        return -1
    if not parser.link(muxer):
        logger.error("Failed to link parser to muxer")
        return -1
    if not muxer.link(filesink):
        logger.error("Failed to link muxer to filesink")
        return -1

    # Create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    nvosd_sink_pad = nvosd.get_static_pad("sink")
    if not nvosd_sink_pad:
        logger.error("Failed to get nvosd sink pad")
        return -1

    nvosd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, nvosd_sink_pad_buffer_probe, None)
    GLib.timeout_add_seconds(PERF_MEASUREMENT_INTERVAL_SEC, perf_struct[stream_id].perf_print_callback)

    pipeline.set_state(Gst.State.PAUSED)
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

    return 0

if __name__ == "__main__":
    main()