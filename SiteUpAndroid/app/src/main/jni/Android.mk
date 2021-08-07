LOCAL_PATH := $(call my-dir)

# change this folder path to yours
NCNN_INSTALL_PATH := ${LOCAL_PATH}/ncnn-20201230

include $(CLEAR_VARS)
LOCAL_MODULE := ncnn
LOCAL_SRC_FILES := $(NCNN_INSTALL_PATH)/$(TARGET_ARCH_ABI)/libncnn.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := alg
LOCAL_SRC_FILES := algorithm.cpp config_parse.cpp algs/alg_basic.cpp algs/alg_classify.cpp algs/alg_keypoint.cpp algs/alg_yolo_fastest.cpp algs/alg_yolov5.cpp
LOCAL_C_INCLUDES := $(NCNN_INSTALL_PATH)/include $(NCNN_INSTALL_PATH)/include/ncnn  $(LOCAL_PATH)/algs
LOCAL_STATIC_LIBRARIES := ncnn
LOCAL_CFLAGS := -O2 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_CPPFLAGS := -O2 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_LDFLAGS += -Wl,--gc-sections
LOCAL_CFLAGS += -fopenmp
LOCAL_CPPFLAGS += -fopenmp
LOCAL_LDFLAGS += -static-openmp -fopenmp
LOCAL_LDLIBS := -lz -llog -ljnigraphics -lvulkan -landroid
include $(BUILD_SHARED_LIBRARY)
