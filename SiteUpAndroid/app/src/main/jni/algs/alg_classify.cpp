#include <stdio.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include "alg_classify.h"


int AlgClassify::Init(AAssetManager *mgr, std::string sConfigFile, std::string sParamFile, std::string sBinFile)
{  
	__android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "AlgClassify::Init");
    
    //phase the cfg info
    m_sConfigFile = sConfigFile;
    m_nOutBlob = GetConfigInt(m_sConfigFile, "ncnn", "out_index");
    m_vecAlgShape = GetConfigIntVec(m_sConfigFile, "ncnn", "input_shape");
    this->FloatVecToArray(m_arrMean, GetConfigFloatVec(m_sConfigFile, "ncnn", "mean"));
    this->FloatVecToArray(m_arrStd, GetConfigFloatVec(m_sConfigFile, "ncnn", "std"));
    
    //show the cfg info
    _ShowCfgInfos();
    
    //load param and bin file
    this->LoadParamAndBin(m_cNet, mgr, sParamFile, sBinFile);
    
    return 0;
}

void * AlgClassify::Run(const unsigned char * pImage, int nColorType, int nHeight, int nWidth, int nNumThreads)
{
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "AlgClassify::Run");
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "image color type: %d", nColorType);
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "image height:  %d", nHeight);
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "image width:  %d", nWidth);
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "threads num:  %d", nNumThreads);
        
    //create the extractor
    ncnn::Extractor cNetEx = m_cNet.create_extractor();
    cNetEx.set_num_threads(nNumThreads);

    //create input
    ncnn::Mat cInput = _ImageProcess(pImage, nColorType, nHeight, nWidth);
    int nRet = cNetEx.input(0, cInput);
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "set input data finished. return %d", nRet);
    
    //create output
    ncnn::Mat cOut;
    cNetEx.extract(m_nOutBlob, cOut);
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "classify finished");
    
    //package the result
    _PackageResult(cOut);
    
    return m_pResultAddr;
}

void AlgClassify::Fini()
{
    std::cout << std::endl << "AlgClassify::Fini" << std::endl;
    return;
}

void AlgClassify::_ShowCfgInfos()
{
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "cfg_file: %s", m_sConfigFile.c_str());
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "out_blob: %d", m_nOutBlob);
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "input_shape: ");
    for(int i=0; i<m_vecAlgShape.size(); i++)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", " %d", m_vecAlgShape[i]);
    }
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "mean: %.4f %.4f %.4f", m_arrMean[0], m_arrMean[1], m_arrMean[2]);
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "std: %.4f %.4f %.4f", m_arrStd[0], m_arrStd[1], m_arrStd[2]);
}

ncnn::Mat AlgClassify::_ImageProcess(const unsigned char * pImage, int nColorType, int nHeight, int nWidth)
{
    //create input
    int nColorTrans = 0;
    if(nColorType == IMG_COLOR_BGR) nColorTrans = ncnn::Mat::PIXEL_BGR;
    if(nColorType == IMG_COLOR_RGB) nColorTrans = ncnn::Mat::PIXEL_RGB2BGR;
    if(nColorType == IMG_COLOR_RGBA) nColorTrans = ncnn::Mat::PIXEL_RGBA2BGR;
    std::cout << "color trans: " << nColorTrans << std::endl;
    ncnn::Mat cInput = ncnn::Mat::from_pixels_resize(pImage,
                                                        nColorTrans,
                                                        nWidth,
                                                        nHeight,
                                                        m_vecAlgShape[0],
                                                        m_vecAlgShape[1]);
                                                        
    
    cInput.substract_mean_normalize(m_arrMean, m_arrStd);
    return cInput;
}

void AlgClassify::_PackageResult(ncnn::Mat &cOut)
{
    UpdateReultCache(4 + cOut.w);
    int nCls = 0;
    float fProb = 0.0;
    float * pRet = (float *)m_pResultAddr + 4;
    for(int i=0; i < cOut.w; i++)
    {
        pRet[i] = cOut[i];
        if (float(cOut[i]) > fProb)
        {
            fProb = cOut[i];
            nCls = i;
        }
    }
    pRet = (float *)m_pResultAddr;
    pRet[0] = 0.0;
    pRet[1] = float(nCls);
    pRet[2] = fProb;
    pRet[3] = float(cOut.w);
}

