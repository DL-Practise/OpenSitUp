#include <stdio.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include "alg_keypoint.h"


int AlgKeyPoint::Init(AAssetManager *mgr, std::string sConfigFile, std::string sParamFile, std::string sBinFile)
{  
    std::cout << std::endl << "AlgKeyPoint::Init" << std::endl;
    
    //phase the cfg info
    m_sConfigFile = sConfigFile;
    m_nOutBlob = GetConfigInt(m_sConfigFile, "ncnn", "out_index");
    m_vecAlgShape = GetConfigIntVec(m_sConfigFile, "ncnn", "input_shape");
    this->FloatVecToArray(m_arrMean, GetConfigFloatVec(m_sConfigFile, "ncnn", "mean"));
    this->FloatVecToArray(m_arrStd, GetConfigFloatVec(m_sConfigFile, "ncnn", "std"));
    
    // create the argmax layer
    m_layerArgMax = ncnn::create_layer("ArgMax");
    ncnn::ParamDict pd;
    pd.set(0, 1); //output max: True
    pd.set(1, 1); //topk: 1
    m_layerArgMax->load_param(pd);

    //show the cfg info
    _ShowCfgInfos();
    
    //load param and bin file
    this->LoadParamAndBin(m_cNet, mgr, sParamFile, sBinFile);
    
    return 0;
}

void * AlgKeyPoint::Run(const unsigned char * pImage, int nColorType, int nHeight, int nWidth, int nNumThreads)
{
    std::cout << std::endl << "AlgKeyPoint::Run" << std::endl;
    std::cout << "image color type: " << nColorType << std::endl;
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

    //argmax process per channel
    std::vector<ncnn::Mat> vecResult;
    for(int c=0; c<cOut.c; c++)
    {
            ncnn::Mat cResult; 
            m_layerArgMax->forward(cOut.channel(c) , cResult,  m_cNet.opt);
            vecResult.push_back(cResult);

            std::cout << "**** channel : " << c << std::endl;
            std::cout << "dim: " << cResult.dims << std::endl;
            std::cout << "w: " << cResult.w << std::endl;
            std::cout << "h: " << cResult.h << std::endl;
    }


    std::cout << "AlgKeyPoint finished" << std::endl;
    

    //package the result
    _PackageResult(vecResult, cOut.w, cOut.h);
    
    
    return m_pResultAddr;
}

void AlgKeyPoint::Fini()
{
    std::cout << std::endl << "AlgKeyPoint::Fini" << std::endl;
    return;
}

void AlgKeyPoint::_ShowCfgInfos()
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

ncnn::Mat AlgKeyPoint::_ImageProcess(const unsigned char * pImage, int nColorType, int nHeight, int nWidth)
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

void AlgKeyPoint::_PackageResult(std::vector<ncnn::Mat> &vecResult, int feature_w, int feature_h)
{
    UpdateReultCache(2 + vecResult.size() * 3);
    ((float *)m_pResultAddr)[0] = 2.0;
    ((float *)m_pResultAddr)[1] = float(vecResult.size());

    float * pRet = (float *)m_pResultAddr + 2;
    for (int i=0; i<vecResult.size(); i++)
    {
        ncnn::Mat heatmap_max = vecResult[i];
        float score = heatmap_max[0];
        int pos_x = (int)(heatmap_max[1]) % feature_w;
        int pos_y = (int)(heatmap_max[1]) / feature_w;
        float x = (float)pos_x / feature_w;
        float y = (float)pos_y / feature_h;
        std::cout << "**** kp : " << i << std::endl;
        std::cout << "feature_w: " << feature_w << std::endl;
        std::cout << "feature_h: " << feature_h << std::endl;
        std::cout << "score: " << score << std::endl;
        std::cout << "pos_x: " << pos_x << std::endl;
        std::cout << "pos_y: " << pos_y << std::endl;
        std::cout << "x: " << x << std::endl;
        std::cout << "y: " << y << std::endl;
        std::cout << "****" <<  std::endl;
        pRet[3*i + 0] = score;
        pRet[3*i + 1] = x;
        pRet[3*i + 2] = y;
    }
}
