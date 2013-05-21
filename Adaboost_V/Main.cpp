#include"AdaBoost.h"
#include<fstream>
#include<sstream>

using namespace std;

int main(int argc, char ** argv){
	if(argc<=2){
		cerr<<"参数错误：[data.txt] [config.txt]"<<endl;
		exit(-1);
	}
	string datapath = argv[1];
	string configpath = argv[2];
	ifstream config_in(configpath);
	ifstream data_in(datapath);
	Data_Vec alldata;
	Data_Vec traindata;
	Data_Vec testdata;
	AdaBoost *adaboost;
	int ID;
	int act_type;
	int test_start;
	int test_end;
	int weaknum;
	string line;
	while(getline(data_in,line)){
		istringstream in(line);
		in>>ID;
		ID-=1;
		in>>act_type;
		Data tmp;
		if(act_type<8){
			tmp.label = 1;
		}
		else{
			tmp.label = -1;
		}
		double dtmp;
		while(in>>dtmp){
			tmp.features.push_back(dtmp);
		}
		alldata.push_back(tmp);
	}
	while (config_in>>test_start>>test_end>>weaknum)
	{
		traindata.clear();
		testdata.clear();
		for (int i = test_start; i <= test_end; i++)
		{
			testdata.push_back(alldata[i]);
		}
		for (int i = 0; i < alldata.size(); i++)
		{
			if(i<test_start || i>test_end)
				traindata.push_back(alldata[i]);
		}
		adaboost = new AdaBoost(traindata,testdata,weaknum);
		cout<<"使用"<<weaknum<<"个弱分类器："<<endl;
		cout<<"Start training....."<<endl;
		adaboost->train();
		cout<<"Train complete !"<<endl;
		cout<<"Testing........"<<endl;
		cout<<"正确率:";
		cout<<adaboost->test()*100;
		cout<<"%"<<endl<<endl;
		//adaboost->OutputDetail();
		delete adaboost;
	}
	system("pause");
	return 0;
}