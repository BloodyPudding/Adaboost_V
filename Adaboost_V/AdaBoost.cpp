#include"AdaBoost.h"




bool node_cmp(Node a,Node b){
	return a.value<b.value;
}

AdaBoost::AdaBoost(Data_Vec TrainData,Data_Vec TestData,int Weak_Classifier_Num){
	this->TrainData = TrainData;
	this->TestData = TestData;
	this->TrainDataNum = TrainData.size();
	this->Dimension = TrainData[0].features.size();
	this->TestDataNum = TestData.size();
	this->Weak_Classifier_Num = Weak_Classifier_Num;
	for (int i = 0; i < Dimension; i++)
	{
		is_dimension_used.push_back(false);
	}
}

void AdaBoost::train(){
	Strong_Classifier sc;
	for( int i = 0; i <TrainDataNum; i++)
		weight.push_back(1.0/TrainDataNum);
	for( int turn = 0; turn < Weak_Classifier_Num; turn++){
		Weak_Classifier wc = weak_train();
		sc.weak.push_back(wc);
		double err = wc.error;
		if( err == 1)
			err = 0;
		else if(err == 0)
			err = eps;
		double beta = err;//(1.0-err);
		//sc.weight.clear();
		sc.weight.push_back(log(1.0/beta)/Log10);
		double sumWeight = 0;
		for(int i = 0; i < TrainDataNum; i++){
			int tlabel;
			if( TrainData[i].features[wc.Dimension_ID] > wc.value){
				tlabel = 1;
			}
			else{
				tlabel = -1;
			}
			if(tlabel == TrainData[i].label)
				weight[i] = weight[i]*beta;
		}
		for(int i = 0; i < TrainDataNum; i++)
			sumWeight +=weight[i];
		for(int i = 0; i < TrainDataNum; i++)
			weight[i]/=sumWeight;
	}
	double Threshold= 0;
	std::vector<Node> buf;
	for (int i = 0; i < TrainDataNum; i++)
	{
		Node tnode ;
		tnode.value = 0;
		tnode.ID = i;
		tnode.label = TrainData[i].label;
		for (int j= 0; j < sc.weak.size(); j++)
		{
			if( TrainData[i].features[sc.weak[j].Dimension_ID] >= sc.weak[j].value){
				tnode.value+=sc.weight[j];
			}
		}
		buf.push_back(tnode);
	}
	sort(buf.begin(),buf.end(),node_cmp);
	double min_error = TrainDataNum+1;
	for (int i = 0; i <=TrainDataNum; i++)
	{
		double nowthr = 0;
		double nowerr = 0;
		if(i==0){
			nowthr = buf[i].value-0.05;
		}
		else if(i==TrainDataNum){
			nowthr = buf[i-1].value+0.05;
		}
		else{
			nowthr = (buf[i].value+buf[i-1].value)/2.0;
		}
		for (int j = 0; j < TrainDataNum; j++)
		{
			if(buf[j].value < nowthr && buf[j].label == 1){
				nowerr +=1;
			}
			if(buf[j].value >= nowthr && buf[j].label == -1){
				nowerr +=1;
			}
		}
		nowerr/=TrainDataNum;
		if(nowerr<min_error){
			sc.Threshold= nowthr;
			min_error = nowerr;
		}
	}
	strong_c =  sc;
}

Weak_Classifier AdaBoost::weak_train(){
	Weak_Classifier wm ;
	double keepThr = 0;
	int keeplabel = 0;
	int keepDim  = 0;
	double errorWeight = 2.0;
	std::vector<Node> sortbuf;
	for(int dim = 0; dim< Dimension; dim++){
		sortbuf.clear();
		for(int i = 0; i < TrainDataNum; i++){
			Node t;
			t.value = TrainData[i].features[dim];
			t.label = TrainData[i].label;
			t.ID = i;
			sortbuf.push_back(t);
		}
		std::sort(sortbuf.begin(),sortbuf.end(),node_cmp);

		double posErrorWeight = 0;
		double nowThr;

		for(int i = 0; i<= TrainDataNum; i++){
			if( i == 0 ){
				nowThr = sortbuf[0].value - 0.5;
				for( int j = 0; j< TrainDataNum; j++ ) 
					if ( sortbuf[j].label  ==-1 )
						posErrorWeight += weight[sortbuf[j].ID];
			}
			else {
				if ( i ==  TrainDataNum )
					nowThr = sortbuf[TrainDataNum-1].value+0.5;
				else if ( sortbuf[i].value == sortbuf[i-1].value )
					continue;
				else
					nowThr = (sortbuf[i].value  + sortbuf[i-1].value )/2;
				posErrorWeight += weight[sortbuf[i-1].ID]*sortbuf[i-1].label;
				int  k = i - 1;
				while(k-1>=0 && sortbuf[k-1].value  == sortbuf[k].value ){
					k--;
					posErrorWeight += weight[sortbuf[k].ID]*sortbuf[k].label;
				}
			}

			if( posErrorWeight < errorWeight && !is_dimension_used[dim]){
				errorWeight = posErrorWeight;
				keeplabel = 1;
				keepThr = nowThr;
				keepDim = dim;
			}
			/*else if( 1.0 - posErrorWeight < errorWeight && !is_dimension_used[dim]){
				errorWeight = 1.0 - posErrorWeight;
				keeplabel = -1;
				keepThr = nowThr;
				keepDim = dim;
			}*/
		}
	}
	if(errorWeight<0){
		errorWeight=0;
	}
	wm.value = keepThr;
	wm.label = keeplabel;
	wm.Dimension_ID = keepDim;
	wm.error = errorWeight;
	is_dimension_used[keepDim]= true;
	return wm;
}

double AdaBoost::test(){
	std::vector<Node> buf;
	for (int i = 0; i < TestDataNum; i++)
	{
		Node tnode ;
		tnode.value = 0;
		tnode.ID = i;
		tnode.label = TestData[i].label;
		for (int j= 0; j < strong_c.weak.size(); j++)
		{
			if( TestData[i].features[strong_c.weak[j].Dimension_ID] > strong_c.weak[j].value){
				tnode.value+=strong_c.weight[j];
			}
		}
		buf.push_back(tnode);
	}
	double error=0;
	for (int i = 0; i < buf.size(); i++)
	{
		int tlabel = 0;
		if(buf[i].value < strong_c.Threshold && buf[i].label == 1){
			error+=1;
		}
		if(buf[i].value >= strong_c.Threshold && buf[i].label == -1){
			error +=1;
		}
	}
	error/=TestDataNum;
	return 1-error;
}

void AdaBoost::OutputDetail(){
	std::cout<<"Strong Classifier Threshold:"<<strong_c.Threshold<<std::endl;
	for (int i = 0; i < strong_c.weak.size(); i++)
	{
		std::cout<<"Weak Classifier "<<(i+1)<<":"<<strong_c.weak[i].value<<"\t"
			<<strong_c.weak[i].error<<"\t"<<strong_c.weak[i].Dimension_ID<<"\t"
			<<strong_c.weak[i].label<<"\t"<<strong_c.weight[i]<<std::endl;
	}
}