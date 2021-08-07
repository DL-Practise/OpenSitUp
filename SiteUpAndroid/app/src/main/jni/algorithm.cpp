#include <stdio.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include "algorithm.h"
#include "alg_basic.h"
#include "alg_classify.h"
#include "alg_yolo_fastest.h"
#include "alg_yolov5.h"
#include "alg_keypoint.h"

AlgBasic * pAlg = NULL;
int nGlobalTHreadNum = 1;

jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "JNI_OnLoad");
    return JNI_VERSION_1_4;
}

void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "JNI_OnUnload");
}

jboolean Java_com_zx_cnn_Alg_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "Java_com_tencent_cnn_Alg_Init");
    char *szConfigFile = "[ncnn]\n \
                          model_type = keypoint\n \
                          #w and h for input\n \
                          input_shape = 224,224\n \
                          out_index = 167\n \
                          mean = 103.530000,116.280000,123.675000\n \
                          std = 0.017429,0.017507,0.017125";

    char *szParamFile = "model_24.param.bin";
    char *szBinFile = "model_24.bin";

    if( KeyExist(szConfigFile, "ncnn", "infer_threads") )
    {
        nGlobalTHreadNum = GetConfigInt(szConfigFile, "ncnn", "infer_threads");
        __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "infer_threads exist value is %d", nGlobalTHreadNum);
    }
    else
    {
        __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "infer_threads not exist use default 1");
    }

    std::string model_type = GetConfigString(szConfigFile, "ncnn", "model_type");
    if (model_type == "classify")
    {
        __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "model type is : classify");
        pAlg = new AlgClassify();
        
    }
    else if (model_type == "detection-yolo_fastest")
    {
        __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "model type is : detection-yolo_fastest");
        pAlg = new AlgYoloFastest();
    }
    else if (model_type == "detection-yolov5")
    {
        __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "model type is : detection-yolov5");
        pAlg = new AlgYoloV5();
    }
    else if (model_type == "keypoint")
    {
        __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "model type is : keypoint");
        pAlg = new AlgKeyPoint();
    }
    else
    {
        __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "error: unknow model type : %s", model_type.c_str());
    }

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    return pAlg->Init(mgr, szConfigFile, szParamFile, szBinFile);
}

jfloatArray Java_com_zx_cnn_Alg_Run(JNIEnv* env, jobject thiz, jobject bitmap)
{
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "Java_com_tencent_cnn_Alg_Run");
    //trans bitmap to char *
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return 0;
    void* indata = NULL;
    AndroidBitmap_lockPixels(env, bitmap, &indata);
    float * pRet = (float *)(pAlg->Run((const unsigned char *) indata, IMG_COLOR_RGBA, height, width, nGlobalTHreadNum));
	AndroidBitmap_unlockPixels(env, bitmap);

	//package the return_array
    int alg_type = int(pRet[0]);
    int nTotalNum = 0;
    if(alg_type == 0)
    {
        int nClassNum = int(pRet[3]);
        nTotalNum = nClassNum+4;
    }

    if (alg_type == 1)
    {
        int nBoxNum = int(pRet[1]);
        nTotalNum = nBoxNum*6 + 2;
    }

    if (alg_type == 2)
    {
        int nKpNum = int(int(pRet[1]));
        nTotalNum = nKpNum*3 + 2;
    }

    jfloatArray array = env-> NewFloatArray(nTotalNum);
    jfloat temp_array[nTotalNum];
    for(int i=0; i<nTotalNum; i++)
    {
        temp_array[i] = pRet[i];
    }

    env->SetFloatArrayRegion(array, 0, nTotalNum, temp_array);
    return array;
}

void Java_com_zx_cnn_Alg_Fini()
{
    return pAlg->Fini();
}

