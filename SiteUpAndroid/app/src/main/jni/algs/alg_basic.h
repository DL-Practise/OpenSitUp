#pragma once
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include "algorithm.h"

class AlgBasic
{
public:
    AlgBasic();
    ~AlgBasic();
    virtual int Init(AAssetManager *mgr, std::string sConfigFile, std::string sParamFile, std::string sBinFile) = 0;
    virtual void *Run(const unsigned char * pImage, int nColorType, int nHeight, int nWidth, int nNumThreads) = 0;
    virtual void Fini() = 0;
    void FloatVecToArray(float dst[], std::vector<float> src);
    ncnn::Mat FloatVecToMat(std::vector<float> src);
    void LoadParamAndBin(ncnn::Net &net, AAssetManager *mgr, std::string sParamFile, std::string sBinFile);
    void UpdateReultCache(int nElemCount);
    std::vector<std::string> split(std::string str, std::string pattern);

protected:
    void *             m_pResultAddr;
    int                m_nResultCount;
};
