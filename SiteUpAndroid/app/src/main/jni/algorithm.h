#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__
#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include "net.h"
#include "config_parse.h"


//color define
#define IMG_COLOR_BGR  0
#define IMG_COLOR_RGB  11
#define IMG_COLOR_RGBA 12


extern "C" JNIEXPORT  jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved);

extern "C" JNIEXPORT  void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved);

//brief: 算法库初始化，申请相关资源
//args：
//  char *szConfigFile：配置文件
//  char *szParamFile： 网络结构文件
//  char *szBinFile：   参数文件
//return:
//  0:   成功
//  非0：错误号
extern "C" JNIEXPORT jboolean JNICALL Java_com_zx_cnn_Alg_Init(JNIEnv* env, jobject thiz, jobject assetManager);

//brief: 算法库运行
//args：
//  const unsigned char *  pImage：彩色图（h*w*c）
//  int                    图片的色彩类型
//  int nImgHeight：       图像的高度
//  int nImgWidth：        图像的宽度
//  int nNumThreads:       并行的线程数目
//return(float 数组):
//  return[0]: 算法类型(0:分类; 1:检测; 2:关键点 >10:特定类型)
//  如果是分类（0）
//    return[1]: top1的类别
//    return[2]: top1的概率
//    return[3]: 分类的类别数目
//    return[4]: cls1的概率
//    return[5]: cls2的概率
//    ...      : ...
//  如果是检测（1）
//    return[1]: 目标框的个数
//    return[2-7]: 第1个框[cls,prob,x1,y1,x2,y2]的信息
//    return[8-13]: 第2个框[cls,prob,x1,y1,x2,y2]的信息
//    ...         : ...
//  如果是关键点（2）
//    return[1]: 关键点的个数
//    return[2-4]: 第1个关键点的信息(score, x,y)
//    return[5-7]: 第2个关键点的信息(score, x,y)
//    ...         : ...
extern "C" JNIEXPORT jfloatArray JNICALL Java_com_zx_cnn_Alg_Run(JNIEnv* env, jobject thiz, jobject bitmap);

//brief: 算法库反初始化，释放相关资源
extern "C" JNIEXPORT void JNICALL Java_com_zx_cnn_Alg_Fini();

#endif

