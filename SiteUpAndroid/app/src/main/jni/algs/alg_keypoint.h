#pragma once
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include "alg_basic.h"


class AlgKeyPoint: public AlgBasic
{
public:
    int Init(AAssetManager *mgr, std::string sConfigFile, std::string sParamFile, std::string sBinFile);
    void *Run(const unsigned char * pImage, int nColorType, int nHeight, int nWidth, int nNumThreads);
    void Fini();
    void _ShowCfgInfos();
    ncnn::Mat _ImageProcess(const unsigned char * pImage, int nColorType, int nHeight, int nWidth);
    void _PackageResult(std::vector<ncnn::Mat> &vecResult, int feature_w, int feature_h);
    
private:
    std::string        m_sConfigFile;
    ncnn::Net          m_cNet;
    int                m_nOutBlob;
    std::vector<int>   m_vecAlgShape;
	float              m_arrMean[3];
    float              m_arrStd[3];
    ncnn::Layer       *m_layerArgMax;
};
