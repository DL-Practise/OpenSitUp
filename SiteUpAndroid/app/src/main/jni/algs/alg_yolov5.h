#pragma once
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include "alg_basic.h"

#define FLT_MAX 3.402823466e+38F 

class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus();
    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const;
};

class CvRect
{
    public:
        CvRect();
        ~CvRect();
        float area();
        std::vector<float> x0y0x1y1();
    public:
        float x;
        float y;
        float width;
        float height;
};

struct Object
{
    CvRect rect;
    int label;
    float prob;
};

float intersection_area(Object& a, Object& b);

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);

void qsort_descent_inplace(std::vector<Object>& faceobjects);

void nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);

float sigmoid(float x);

void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);

class AlgYoloV5: public AlgBasic
{
public:
    int Init(AAssetManager *mgr, std::string sConfigFile, std::string sParamFile, std::string sBinFile);
    void *Run(const unsigned char * pImage, int nColorType, int nHeight, int nWidth, int nNumThreads);
    void Fini();
    void _ShowCfgInfos();
    ncnn::Mat _ImageProcess(const unsigned char * pImage, int nColorType, int nHeight, int nWidth, int &wpad, int &hpad, float &scale);
    void _PackageResult(std::vector<Object> &proposals, std::vector<int> &picked, int img_w, int img_h, int wpad, int hpad, float scale);
    
private:
    std::string               m_sConfigFile;
    ncnn::Net                 m_cNet;
    std::vector<std::string>  m_vecOutBlobs;
    int                       m_nMaxImgSide;
	float                     m_arrMean[3];
    float                     m_arrStd[3];
    ncnn::Mat                 m_matAnchor8;
    ncnn::Mat                 m_matAnchor16;
    ncnn::Mat                 m_matAnchor32;

    float                     m_fProbThres;
    float                     m_fNmsThres;
};
