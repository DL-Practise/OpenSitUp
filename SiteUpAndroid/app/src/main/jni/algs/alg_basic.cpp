#include <stdio.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include "alg_basic.h"

AlgBasic::AlgBasic()
{
    m_pResultAddr = NULL;
    m_nResultCount = 0;
}

AlgBasic::~AlgBasic()
{
    if(m_pResultAddr != NULL)
    {
        free(m_pResultAddr);
        m_nResultCount = 0;
    }
}

void AlgBasic::FloatVecToArray(float dst[], std::vector<float> src)
{
    for(int i=0; i<src.size(); i++)
    {
        dst[i] = src[i];
    }
}

ncnn::Mat AlgBasic::FloatVecToMat(std::vector<float> src)
{
    ncnn::Mat temp_mat = ncnn::Mat(src.size());
    for(int i=0; i<src.size(); i++)
    {
        temp_mat[i] = src[i];
    }
    return temp_mat;
}

void AlgBasic::LoadParamAndBin(ncnn::Net &net, AAssetManager *mgr, std::string sParamFile, std::string sBinFile)
{
    string::size_type position = sParamFile.find(".bin");
    if (position != sParamFile.npos)
    {
        int nRet = net.load_param_bin(mgr, sParamFile.c_str());
        __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "load param.bin file(%s) return %d", sParamFile.c_str(), nRet);
    }
    else
    {
        int nRet = net.load_param(mgr, sParamFile.c_str());
        __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "load param file(%s) return %d", sParamFile.c_str(), nRet);
    }
    int nRet = net.load_model(mgr, sBinFile.c_str());
    __android_log_print(ANDROID_LOG_DEBUG, "zhengxing", "load bin file(%s) return %d", sBinFile.c_str(), nRet);
}


void AlgBasic::UpdateReultCache(int nElemCount)
{
    if(m_nResultCount >= nElemCount)
    {
        return ;
    }

    if(m_pResultAddr != NULL)
    {
        free(m_pResultAddr);
        m_nResultCount = 0;
    }

    m_pResultAddr = malloc(nElemCount * sizeof(float));
    m_nResultCount = nElemCount;
    if(m_pResultAddr == NULL)
    {
        std::cout<<"malloc for " << nElemCount << " elements failed" << std::endl;
        m_nResultCount = 0;
    }
    else
    {
        std::cout<<"malloc for " << nElemCount << " elements success" << std::endl;
    }

}

std::vector<std::string> AlgBasic::split(std::string str, std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str += pattern;//扩展字符串以方便操作
    int size = str.size();
    for (int i = 0; i < size; i++)
    {
        pos = str.find(pattern, i);
        if (pos < size)
        {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}