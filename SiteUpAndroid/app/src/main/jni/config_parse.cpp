#include <stdio.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include "config_parse.h"

int KeyExist(std::string sFile, std::string sSection, std::string sName)
{
    CParseIniFile config;
    std::map<string,string> mapInfos;
    config.ReadConfig(sFile, mapInfos, sSection.c_str());
    
    std::map<string, string>::iterator  iter;
    for(iter = mapInfos.begin(); iter != mapInfos.end(); iter++)
    {
        if(iter->first == sName)
        {
           return 1;   
        }
    }
    return 0;
}

std::string GetConfigString(std::string sFile, std::string sSection, std::string sName)
{
    CParseIniFile config;
    std::map<string,string> mapInfos;
    config.ReadConfig(sFile, mapInfos, sSection.c_str());
    
    std::map<string, string>::iterator  iter;
    for(iter = mapInfos.begin(); iter != mapInfos.end(); iter++)
    {
        if(iter->first == sName)
        {
           return iter->second;   
        }
    }
    return std::string("");
}

int GetConfigInt(std::string sFile, std::string sSection, std::string sName)
{
    return atoi(GetConfigString(sFile, sSection,sName).c_str());
}

float GetConfigFloat(std::string sFile, std::string sSection, std::string sName)
{
    return atof(GetConfigString(sFile, sSection,sName).c_str());
}

std::vector<std::string> GetConfigStringVec(std::string sFile, std::string sSection, std::string sName)
{
    std::vector<std::string> sRetVec;
    std::string sInfo = GetConfigString(sFile, sSection, sName);
    istringstream tmp_string(sInfo);
    string sDomain;
    while (getline(tmp_string, sDomain, ','))
    {
        sRetVec.push_back(sDomain);
    }
    return sRetVec;
}

std::vector<int> GetConfigIntVec(std::string sFile, std::string sSection, std::string sName)
{
    std::vector<std::string> sStringVec = GetConfigStringVec(sFile, sSection, sName);
    std::vector<int> sIntVec;
    for(int i=0; i<sStringVec.size(); i++)
    {
        sIntVec.push_back(atoi(sStringVec[i].c_str()));
    }
    return sIntVec;
}

std::vector<float> GetConfigFloatVec(std::string sFile, std::string sSection, std::string sName)
{
    std::vector<std::string> sStringVec = GetConfigStringVec(sFile, sSection, sName);
    std::vector<float> sFloatVec;
    for(int i=0; i<sStringVec.size(); i++)
    {
        sFloatVec.push_back(atof(sStringVec[i].c_str()));
    }
    return sFloatVec;
}

std::vector<std::vector<std::string> > GetConfigStringVec2D(std::string sFile, std::string sSection, std::string sName)
{
    std::vector<std::string> sVec1D;
    std::string sInfo = GetConfigString(sFile, sSection, sName);
    istringstream tmp_string(sInfo);
    string sDomain;
    while (getline(tmp_string, sDomain, ';'))
    {
        sVec1D.push_back(sDomain);
    }

    std::vector<std::vector<std::string> > sVec2D;
    for(int i=0; i<sVec1D.size(); i++)
    {
        std::vector<std::string> sVec;
        istringstream tmp_string(sVec1D[i]);
        string sDomain;
        while (getline(tmp_string, sDomain, ','))
        {
            sVec.push_back(sDomain);
        }
        sVec2D.push_back(sVec);
    }
    
    return sVec2D;
}
      
std::vector<std::vector<int> > GetConfigIntVec2D(std::string sFile, std::string sSection, std::string sName)
{
    std::vector<std::vector<std::string> > sVecString2D = GetConfigStringVec2D(sFile, sSection, sName);
    std::vector<std::vector<int> > sVecInt2D;
    
    for(int i=0; i<sVecString2D.size(); i++)
    {
        std::vector<int> vecTemp;
        for (int j=0; j<sVecString2D[i].size(); j++)
        {
            vecTemp.push_back(atoi(sVecString2D[i][j].c_str()));
        }
        sVecInt2D.push_back(vecTemp);
    }
    return sVecInt2D;
}

std::vector<std::vector<float> > GetConfigFloatVec2D(std::string sFile, std::string sSection, std::string sName)
{
    std::vector<std::vector<std::string> > sVecString2D = GetConfigStringVec2D(sFile, sSection, sName);
    std::vector<std::vector<float> > sVecFloat2D;
    
    for(int i=0; i<sVecString2D.size(); i++)
    {
        std::vector<float> vecTemp;
        for (int j=0; j<sVecString2D[i].size(); j++)
        {
            vecTemp.push_back(atof(sVecString2D[i][j].c_str()));
        }
        sVecFloat2D.push_back(vecTemp);
    }
    return sVecFloat2D;
}
      
      
CParseIniFile::CParseIniFile()
{

}

CParseIniFile::~CParseIniFile()
{

}

std::vector<std::string> CParseIniFile::split_string(const std::string& str, const std::string& delimiter)
{
    //from ncnn-android-squeezenet-master
    std::vector<std::string> strings;
    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

bool CParseIniFile::ReadConfig(const string& filename, map<string, string>& mContent, const char* section)
{
    mContent.clear();
    ifstream infile(filename.c_str());
    //the input is cfg string
    if (!infile)
    {
        //printf("readconfg from cfg string\n");
        std::vector<std::string> cfg_lines = split_string(filename, "\n");
        //std::cout << ">>> the ori string: " << filename << std::endl;
        //for(int i=0; i < cfg_lines.size(); i++)
        //{
        //    std::cout << ">>> " << cfg_lines[i] << std::endl;
        //}
        
        string line, key, value;
        int pos = 0;
        string Tsection = string("[") + section + "]";
        bool flag = false;
        int i=0;
        while(i < cfg_lines.size())
        {
            std::string line = cfg_lines[i];
            if (!flag)
            {
                pos = line.find(Tsection, 0);
                if (-1 == pos)
                {
                    continue;
                }
                else
                {
                    flag = true;
                    line = cfg_lines[i+1];
                    i++;
                }
            }
            if (0 < line.length() && '[' == line.at(0))
            {
                break;
            }
            if (0 < line.length() && AnalyseLine(line, key, value))
            {

                if (value.length() > 0)
                {
                    if (value[value.size() - 1] == '\r')
                    {
                        value[value.size() - 1] = '\0';
                    }
                }
                mContent[key] = value;
            }
            i++;
        }
        return true;
    }
    //the input is cfg file path
    else
    {
        //printf("readconfg from cfg file\n");
        string line, key, value;
        int pos = 0;
        string Tsection = string("[") + section + "]";
        bool flag = false;
        while (getline(infile, line))
        {
            if (!flag)
            {
                pos = line.find(Tsection, 0);
                if (-1 == pos)
                {
                    continue;
                }
                else
                {
                    flag = true;
                    getline(infile, line);
                }
            }
            if (0 < line.length() && '[' == line.at(0))
            {
                break;
            }
            if (0 < line.length() && AnalyseLine(line, key, value))
            {

                if (value.length() > 0)
                {
                    if (value[value.size() - 1] == '\r')
                    {
                        value[value.size() - 1] = '\0';
                    }
                }
                mContent[key] = value;
            }
        }
        infile.close();
        return true;
    }
}

bool CParseIniFile::AnalyseLine(const string & line, string & key, string & val)
{
    if (line.empty())
    {
        return false;
    }
    int start_pos = 0, end_pos = line.size() - 1, pos = 0;
    if ((pos = line.find(COMMENT_CHAR)) != -1)
    {
        if (0 == pos)
        {//行的第一个字符就是注释字符
            return false;
        }
        end_pos = pos - 1;
    }
    string new_line = line.substr(start_pos, start_pos + 1 - end_pos);  // 预处理，删除注释部分

    if ((pos = new_line.find('=')) == -1)
    {
        return false;  // 没有=号
    }

    key = new_line.substr(0, pos);
    val = new_line.substr(pos + 1, end_pos + 1 - (pos + 1));

    Trim(key);
    if (key.empty())
    {
        return false;
    }
    Trim(val);
    return true;
}

void CParseIniFile::Trim(string & str)
{
    if (str.empty())
    {
        return;
    }
    int i, start_pos, end_pos;
    for (i = 0; i < str.size(); ++i)
    {
        if (!IsSpace(str[i]))
        {
            break;
        }
    }
    if (i == str.size())
    { //全部是空白字符串
        str = "";
        return;
    }

    start_pos = i;

    for (i = str.size() - 1; i >= 0; --i)
    {
        if (!IsSpace(str[i]))
        {
            break;
        }
    }
    end_pos = i;
    str = str.substr(start_pos, end_pos - start_pos + 1);
}

bool CParseIniFile::IsSpace(char c)
{
    if (' ' == c || '\t' == c)
    {
        return true;
    }
    return false;
}

bool CParseIniFile::IsCommentChar(char c)
{
    switch (c)
    {
    case COMMENT_CHAR:
        return true;
    default:
        return false;
    }
}

void CParseIniFile::PrintConfig(const map<string, string> & mContent)
{
    map<string, string>::const_iterator mite = mContent.begin();
    for (; mite != mContent.end(); ++mite)
    {
        cout << mite->first << "=" << mite->second << endl;
    }
}

