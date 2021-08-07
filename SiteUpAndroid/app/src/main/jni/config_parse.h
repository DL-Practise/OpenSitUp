#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <vector>
using namespace std;
#define COMMENT_CHAR '#'

int KeyExist(std::string sFile, std::string sSection, std::string sName);
std::string GetConfigString(std::string sFile, std::string sSection, std::string sName);
int GetConfigInt(std::string sFile, std::string sSection, std::string sName);
float GetConfigFloat(std::string sFile, std::string sSection, std::string sName);

std::vector<std::string> GetConfigStringVec(std::string sFile, std::string sSection, std::string sName);
std::vector<int> GetConfigIntVec(std::string sFile, std::string sSection, std::string sName);
std::vector<float> GetConfigFloatVec(std::string sFile, std::string sSection, std::string sName);

std::vector<std::vector<std::string> > GetConfigStringVec2D(std::string sFile, std::string sSection, std::string sName);
std::vector<std::vector<int> > GetConfigIntVec2D(std::string sFile, std::string sSection, std::string sName);
std::vector<std::vector<float> > GetConfigFloatVec2D(std::string sFile, std::string sSection, std::string sName);

class CParseIniFile
{
public:
    CParseIniFile();
    ~CParseIniFile();
    std::vector<std::string> split_string(const std::string& str, const std::string& delimiter);
    bool ReadConfig(const string& filename, map<string, string>& mContent, const char* section);
    bool AnalyseLine(const string & line, string & key, string & val);
    void Trim(string & str);
    bool IsSpace(char c);
    bool IsCommentChar(char c);
    void PrintConfig(const map<string, string> & mContent);
private:
};

