#ifndef ADABOOST_H
#define ADABOOST_H

#include<iostream>
#include<vector>
#include<algorithm>
#include<cmath>

const double Log10 = log(10.0);
#define eps 0.00000001

struct Data{
	std::vector<double> features;
	int label;
};

struct Node{
	double value;
	int label;
	int ID;
};

struct Weak_Classifier{
	double value;
	int label;
	int Dimension_ID;
	double error;
};

struct Strong_Classifier{
	double Threshold;
	std::vector<Weak_Classifier> weak;
	std::vector<double> weight;
};

typedef std::vector<Data> Data_Vec;

class AdaBoost{

public :
	Data_Vec TrainData;
	Data_Vec TestData;
	Strong_Classifier strong_c;
	void train();
	Weak_Classifier weak_train();
	std::vector<double> weight;
	std::vector<bool> is_dimension_used;
	double test();
	void OutputDetail();
	AdaBoost(Data_Vec TrainData,Data_Vec TestData,int Weak_Classifier_Num);
	~AdaBoost(){};

private:
	int Weak_Classifier_Num;
	int Dimension;
	int TrainDataNum;
	int TestDataNum;
};


#endif