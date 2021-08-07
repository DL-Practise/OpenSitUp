#include <stdio.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include "alg_yolov5.h"

#define FLT_MAX 3.402823466e+38F 


YoloV5Focus::YoloV5Focus()
{
    one_blob_only = true;
}

int YoloV5Focus::forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int outw = w / 2;
    int outh = h / 2;
    int outc = channels * 4;

    top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outc; p++)
    {
        const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                *outptr = *ptr;

                outptr += 1;
                ptr += 2;
            }

            ptr += w;
        }
    }

    return 0;
}

DEFINE_LAYER_CREATOR(YoloV5Focus)

CvRect::CvRect(){}

CvRect::~CvRect(){}

float CvRect::area()
{
    return width*height;
}

std::vector<float> CvRect::x0y0x1y1()
{
    std::vector<float> cord;
    cord.push_back(x);
    cord.push_back(y);
    cord.push_back(x+width);
    cord.push_back(y+height);
    return cord;
}


float intersection_area(Object& a, Object& b)
{
    std::vector<float> rec1 = a.rect.x0y0x1y1();
    std::vector<float> rec2 = b.rect.x0y0x1y1();

    float left_column_max  = max(rec1[0],rec2[0]);
    float right_column_min = min(rec1[2],rec2[2]);
    float up_row_max       = max(rec1[1],rec2[1]);
    float down_row_min     = min(rec1[3],rec2[3]);

    if (left_column_max >= right_column_min) // or down_row_min <= up_row_max)
    {
        return 0;
    }
    else if (down_row_min <= up_row_max)
    {
        return 0;
    }
    else
    {
        return (down_row_min-up_row_max)*(right_column_min-left_column_max);
    }
}

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}


int AlgYoloV5::Init(AAssetManager *mgr, std::string sConfigFile, std::string sParamFile, std::string sBinFile)
{
    std::cout << std::endl << "AlgYoloV5::Init" << std::endl;
    
    //read and print config info
    m_sConfigFile = sConfigFile;
    m_nMaxImgSide = GetConfigInt(m_sConfigFile, "ncnn", "image_size");
    m_vecOutBlobs = GetConfigStringVec(m_sConfigFile, "ncnn", "out_index");
    this->FloatVecToArray(m_arrMean, GetConfigFloatVec(m_sConfigFile, "ncnn", "mean"));
    this->FloatVecToArray(m_arrStd, GetConfigFloatVec(m_sConfigFile, "ncnn", "std"));
    m_matAnchor8 = this->FloatVecToMat(GetConfigFloatVec(m_sConfigFile, "ncnn", "anchor_1"));
    m_matAnchor16 = this->FloatVecToMat(GetConfigFloatVec(m_sConfigFile, "ncnn", "anchor_2"));
    m_matAnchor32 = this->FloatVecToMat(GetConfigFloatVec(m_sConfigFile, "ncnn", "anchor_3"));
    m_fProbThres = GetConfigFloat(m_sConfigFile, "ncnn", "prob_threshold");
    m_fNmsThres = GetConfigFloat(m_sConfigFile, "ncnn", "nms_threshold");
        
    //show the cfg info
    _ShowCfgInfos();

    // net prepare
    //m_cNet.opt.use_vulkan_compute = true;
    //m_cNet.opt.use_bf16_storage = true;
    m_cNet.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    
    //load param and bin file
    this->LoadParamAndBin(m_cNet, mgr, sParamFile, sBinFile);

    return 0;
}

void * AlgYoloV5::Run(const unsigned char * pImage, int nColorType, int nHeight, int nWidth, int nNumThreads)
{
    std::cout << std::endl << "AlgYoloV5::Run" << std::endl;
    std::cout << "image type: " << nColorType << std::endl;
    std::cout << "image height: " << nHeight << std::endl;
    std::cout << "image width: " << nWidth << std::endl;
    std::cout << "threads num: " << nNumThreads << std::endl;
    
    //create the extractor
    ncnn::Extractor ex = m_cNet.create_extractor();
    ex.set_num_threads(nNumThreads);
    
    //create input
    int wpad, hpad;
    float scale;
    ncnn::Mat in_pad = _ImageProcess(pImage, nColorType, nHeight, nWidth, wpad, hpad, scale);
    ex.input("images", in_pad);
    
    //create output
    std::vector<Object> proposals;
    {
        ncnn::Mat out;
        ex.extract(m_vecOutBlobs[0].c_str(), out);
        std::cout << "network forward in stride 8 finished" << std::endl;
        std::vector<Object> objects8;
        generate_proposals(m_matAnchor8, 8, in_pad, out, m_fProbThres, objects8);
        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        std::cout << "postproc stride 8 finished" << std::endl;
    }
    {
        ncnn::Mat out;
        ex.extract(m_vecOutBlobs[1].c_str(), out);
        std::cout << "network forward in stride 16 finished" << std::endl;
        std::vector<Object> objects16;
        generate_proposals(m_matAnchor16, 16, in_pad, out, m_fProbThres, objects16);
        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        std::cout << "postproc stride 16 finished" << std::endl;
    }
    {
        ncnn::Mat out;
        ex.extract(m_vecOutBlobs[2].c_str(), out);
        std::cout << "network forward in stride 32 finished" << std::endl;
        std::vector<Object> objects32;
        generate_proposals(m_matAnchor32, 32, in_pad, out, m_fProbThres, objects32);
        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        std::cout << "postproc stride 32 finished" << std::endl;
    }
    qsort_descent_inplace(proposals);//sort all proposals by score from highest to lowest
    std::vector<int> picked;//apply nms with nms_threshold
    nms_sorted_bboxes(proposals, picked, m_fNmsThres);
    
    //package the result
    _PackageResult(proposals, picked, nWidth, nHeight, wpad, hpad, scale);
    
    return m_pResultAddr; 
}

void AlgYoloV5::Fini()
{

}

void AlgYoloV5::_ShowCfgInfos()
{
    std::cout << "cfg_file: " << m_sConfigFile << std::endl;
    std::cout << "out blobs: ";
    for(int i=0; i<m_vecOutBlobs.size(); i++)
    {
        std::cout << m_vecOutBlobs[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "mean: " << m_arrMean[0] << "," << m_arrMean[1] << "," << m_arrMean[2] << std::endl;
    std::cout << "std: " << m_arrStd[0] << "," << m_arrStd[1] << "," << m_arrStd[2] << std::endl;   
    std::cout << "image max side: " << m_nMaxImgSide << std::endl;
    
    std::cout << "anchor_8: ";
    for(int i=0; i<m_matAnchor8.total(); i++)
    {
        std::cout << m_matAnchor8[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "anchor_16: ";
    for(int i=0; i<m_matAnchor16.total(); i++)
    {
        std::cout << m_matAnchor16[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "anchor_32: ";
    for(int i=0; i<m_matAnchor32.total(); i++)
    {
        std::cout << m_matAnchor32[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "m_fProbThres: " << m_fProbThres << std::endl;
    std::cout << "m_fNmsThres: " << m_fNmsThres << std::endl;
}

ncnn::Mat AlgYoloV5::_ImageProcess(const unsigned char * pImage, int nColorType, int nHeight, int nWidth, int &wpad, int &hpad, float &scale)
{
    const int target_size = m_nMaxImgSide;
    int img_w = nWidth;
    int img_h = nHeight;

    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
    scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    //create input
    int nColorTrans = 0;
    if(nColorType == IMG_COLOR_BGR) nColorTrans = ncnn::Mat::PIXEL_BGR;
    if(nColorType == IMG_COLOR_RGB) nColorTrans = ncnn::Mat::PIXEL_RGB2BGR;
    if(nColorType == IMG_COLOR_RGBA) nColorTrans = ncnn::Mat::PIXEL_RGBA2BGR;
    std::cout << "color trans: " << nColorTrans << std::endl;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(pImage, nColorTrans, img_w, img_h, w, h);
    std::cout << "resize letter image from(" << img_w << " * " << img_h << ") to (" << w << " * " << h << ")" << std::endl;

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    wpad = (w + 31) / 32 * 32 - w;
    hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    in_pad.substract_mean_normalize(m_arrMean, m_arrStd);
    std::cout << "normalise image" << std::endl;
    
    return in_pad;
}

void AlgYoloV5::_PackageResult(std::vector<Object> &proposals, std::vector<int> &picked, int img_w, int img_h, int wpad, int hpad, float scale)
{
    int count = picked.size();
    UpdateReultCache(2 + count * 6);
    float * pRet = (float *)m_pResultAddr;
    pRet[0] = (float)1.0;
    pRet[1] = (float)count;
    pRet = (float *)m_pResultAddr + 2;
    for(int i=0; i<count; i++)
    {
        // adjust offset to original unpadded
        float x0 = (proposals[picked[i]].rect.x - (wpad / 2)) / scale;
        float y0 = (proposals[picked[i]].rect.y - (hpad / 2)) / scale;
        float x1 = (proposals[picked[i]].rect.x + proposals[picked[i]].rect.width - (wpad / 2)) / scale;
        float y1 = (proposals[picked[i]].rect.y + proposals[picked[i]].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = (std::max)((std::min)(x0, (float)(img_w - 1)), 0.f);
        y0 = (std::max)((std::min)(y0, (float)(img_h - 1)), 0.f);
        x1 = (std::max)((std::min)(x1, (float)(img_w - 1)), 0.f);
        y1 = (std::max)((std::min)(y1, (float)(img_h - 1)), 0.f);
        std::cout << proposals[picked[i]].label << " " << proposals[picked[i]].prob << " " << x0 << " "<< y0 << " " << x1 << " " << y1 << std::endl;

        pRet[i * 6  + 0] = (float)(proposals[picked[i]].label);
        pRet[i * 6  + 1] = (float)(proposals[picked[i]].prob);
        pRet[i * 6  + 2] = (float)x0 / img_w;
        pRet[i * 6  + 3] = (float)y0 / img_h;
        pRet[i * 6  + 4] = (float)x1 / img_w;
        pRet[i * 6  + 5] = (float)y1 / img_h;
        
    }
}