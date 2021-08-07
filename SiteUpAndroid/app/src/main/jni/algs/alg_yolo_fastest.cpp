#include <stdio.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include "alg_yolo_fastest.h"


int AlgYoloFastest::Init(AAssetManager *mgr, std::string sConfigFile, std::string sParamFile, std::string sBinFile)
{
    std::cout << std::endl << "AlgYoloFastest::Init" << std::endl;
    
    //read and print config info
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

void * AlgYoloFastest::Run(const unsigned char * pImage, int nColorType, int nHeight, int nWidth, int nNumThreads)
{
    std::cout << std::endl << "AlgYoloFastest::Run" << std::endl;
    std::cout << "image type: " << nColorType << std::endl;
    std::cout << "image height: " << nHeight << std::endl;
    std::cout << "image width: " << nWidth << std::endl;
    std::cout << "threads num: " << nNumThreads << std::endl;
    
    //create the extractor
    ncnn::Extractor cNetEx = m_cNet.create_extractor();
    cNetEx.set_num_threads(nNumThreads);

    //create input
    ncnn::Mat cInput = _ImageProcess(pImage, nColorType, nHeight, nWidth);
    int nRet = cNetEx.input(0, cInput);
    std::cout << "set input data finished. return " << nRet << std::endl;
    
    //create output
    ncnn::Mat cOut;
    cNetEx.extract(m_nOutBlob, cOut);
    std::cout << "detection finished" << std::endl;
    
    //package the result
    _PackageResult(cOut);
    
    return m_pResultAddr;
}


void AlgYoloFastest::Fini()
{
    std::cout << std::endl << "AlgYoloFastest::Fini" << std::endl;
}

void AlgYoloFastest::_ShowCfgInfos()
{
    std::cout << "cfg_file: " << m_sConfigFile << std::endl;
    std::cout << "out_blob: " << m_nOutBlob << std::endl;
    std::cout << "input_shape: ";
    for(int i=0; i<m_vecAlgShape.size(); i++)
    {
        std::cout << m_vecAlgShape[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "mean: " << m_arrMean[0] << "," << m_arrMean[1] << "," << m_arrMean[2] << std::endl;
    std::cout << "std: " << m_arrStd[0] << "," << m_arrStd[1] << "," << m_arrStd[2] << std::endl;        
}

ncnn::Mat AlgYoloFastest::_ImageProcess(const unsigned char * pImage, int nColorType, int nHeight, int nWidth)
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

void AlgYoloFastest::_PackageResult(ncnn::Mat &cOut)
{
    UpdateReultCache( 2 + cOut.h * cOut.w );
    float * pRet = (float *)m_pResultAddr;
    pRet[0] = 1.0;
    pRet[1] = cOut.h;
    pRet = (float *)m_pResultAddr + 2;
    for(int i=0; i<cOut.h; i++)
    {
        pRet[i * cOut.w + 0] = cOut.row(i)[0];
        pRet[i * cOut.w + 1] = cOut.row(i)[1];
        pRet[i * cOut.w + 2] = cOut.row(i)[2];
        pRet[i * cOut.w + 3] = cOut.row(i)[3];
        pRet[i * cOut.w + 4] = cOut.row(i)[4];
        pRet[i * cOut.w + 5] = cOut.row(i)[5];
    }
}