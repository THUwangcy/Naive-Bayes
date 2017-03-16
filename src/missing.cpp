#include <iostream>
#include <map>
#include <vector>
#include <string.h>
#include <cmath>
#include <fstream>
#include <stdlib.h> 
#include <time.h> 
#include <sstream>
#include "tools.h"

using namespace std;

#define FEATURE_NUM 14 //特征数量
#define LOW50 0
#define HIGH50 1
#define alpha 0.2  //Laplace平滑的系数
#define pi 3.1415926

double TRAIN_RATE = 1; //用于训练的样本比例

double N[2];  //两种类型人群的总人数，0表示≤50K，1表示>50K
double totalNum, totalTrain, totalTest;  //数据总数、训练集总数、测试集总数
vector< vector<string> > samples;  //训练集
map<string, int> features[2][FEATURE_NUM];  //表示某一类别下、某一个特征中该取值在当前类别中的数量，用于取离散值的特征
double u[2][FEATURE_NUM];  //表示某一类别下，某一特征取值的均值
double sita[2][FEATURE_NUM];  //表示某一类别下，某一特征取值的方差，这两个都用于连续取值的特征
bool continuous[FEATURE_NUM] = {1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0};
string mostPossible[2][FEATURE_NUM];  //表示某一类别下，某一离散取值的特征最可能的取值，用于处理缺失值，这里取训练集出现最多的label作最可能
double maxLabel[2][FEATURE_NUM];  //表示某一类别下，某一离散取值的特征出现次数最多的那个label的出现次数
double M[FEATURE_NUM];  //表示某一维特征的最大取值个数（对离散取值而言），用于Laplace平滑
long success = 0;
long correct[2] = {0}, pred[2] = {0}, all[2] = {0};

bool check[40000];

double maxValue[FEATURE_NUM];
double minValue[FEATURE_NUM];
double gap[FEATURE_NUM];
double gapGroup = 20;


void init() {
	memset(N, 0, sizeof(N));
	totalNum = 0; totalTrain = 0; totalTest = 0;
	memset(u, 0, sizeof(u));
	memset(sita, 0, sizeof(sita));
	memset(M, 0, sizeof(M));
	memset(maxLabel, 0, sizeof(maxLabel));
	memset(check, 0, sizeof(check));
	memset(maxValue, 0, sizeof(maxValue));
	for(int i = 0; i < FEATURE_NUM; i ++)  minValue[i] = INT_MAX;
	memset(gap, 0, sizeof(gap));
	srand((unsigned)time(NULL)); 
}

void outputLoadData() {
	cout << "TRAIN_RATE: " << TRAIN_RATE << endl;
	cout << "Traning Scale: " << totalTrain << endl;
	cout << "≤50K: " << N[LOW50] << " >50K: " << N[HIGH50] << endl;
	cout << endl << "maxValue: " << endl;
	for(int j = 0; j < FEATURE_NUM; j ++) {
		if(!continuous[j])  cout << 0 << " ";
		else cout << maxValue[j] << " ";
	}
	cout << endl;
	cout << endl << "minValue: " << endl;
	for(int j = 0; j < FEATURE_NUM; j ++) {
		if(!continuous[j])  cout << 0 << " ";
		else cout << minValue[j] << " ";
	}
	cout << endl;
	cout << endl << "gap: " << endl;
	for(int j = 0; j < FEATURE_NUM; j ++) {
		if(!continuous[j])  cout << 0 << " ";
		else cout << gap[j] << " ";
	}
	cout << endl;
	cout << endl << "mostPossible: " << endl;
	for(int i = 0; i < 2; i ++) {
		for(int j = 0; j < FEATURE_NUM; j ++) {
			cout << mostPossible[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl << "M: " << endl;
	for(int i = 0; i < FEATURE_NUM; i ++) {
		cout << M[i] << " ";
	}
	cout << endl;
	cout << "********LOADING COMPLETE**********" << endl;
}

void loadData() {
	freopen("../data/adult.train", "r", stdin);
	freopen("output", "w", stdout);
	string s; 
	cout << "********LOADING DATA**********" << endl;
	while(cin >> s) {
		vector<string> person = split(s, ',');
		samples.push_back(person);
		totalNum ++;
	}
	for(int k = 0; k < TRAIN_RATE * totalNum; k ++) {
		int index = ((int)rand() % (int)totalNum);
		while(check[index])  index = ((int)rand() % (int)totalNum);
		check[index] = 1;
		vector<string> person = samples[index];
		int curClass = (person[FEATURE_NUM] == "<=50K." ? LOW50 : HIGH50);  //当前读入这个人所属的类别
		N[curClass] ++;   //对应类别总数增加

		//遍历每一维特征，进行相应的处理
		for(int i = 0; i < FEATURE_NUM; i ++) {
			//连续取值特征第一次遍历只算均值
			if(continuous[i] && person[i] != "?") {
				double num = atof(person[i].c_str());
				if(num > maxValue[i])  maxValue[i] = num;
				if(num < minValue[i])  minValue[i] = num;
			}
			else {
				map<string, int>::iterator it = features[curClass][i].find(person[i]);
				//如果已经记录过该特征
				if(it != features[curClass][i].end()) {
					features[curClass][i][person[i]] ++;
					//更新最可能的该特征值
					if(features[curClass][i][person[i]] > maxLabel[curClass][i]) {
						maxLabel[curClass][i] = features[curClass][i][person[i]];
						mostPossible[curClass][i] = person[i];
					}
				}
				else{
					features[curClass][i][person[i]] = 1;
					//如果该特征取值从来没有出现过，增加M
					map<string, int>::iterator it2 = features[1 - curClass][i].find(person[i]);
					if(it2 == features[1 - curClass][i].end())  M[i] ++;
				}
			}
		}
	}
	//计算总数
	totalTrain = N[LOW50] + N[HIGH50];
	for(int i = 0; i < FEATURE_NUM; i ++)  gap[i] = (maxValue[i] - minValue[i]) / gapGroup;
	//计算方差
	for(int i = 0; i < totalTrain; i ++) {
		int curClass = (samples[i][FEATURE_NUM] == "<=50K." ? LOW50 : HIGH50);
		for(int j = 0; j < FEATURE_NUM; j ++) {
			if(!continuous[j] || samples[i][j] == "?")  continue;
			double num = atof(samples[i][j].c_str());
			int index = (num - minValue[j]) / gap[j];
			stringstream ss;
		    string s;
		    ss << index;
		    ss >> s;
		    map<string, int>::iterator it = features[curClass][j].find(s);
			//如果已经记录过该特征
			if(it != features[curClass][j].end()) {
				features[curClass][j][s] ++;
				//更新最可能的该特征值
				if(features[curClass][j][s] > maxLabel[curClass][j]) {
					maxLabel[curClass][j] = features[curClass][j][s];
					mostPossible[curClass][j] = s;
				}
			}
			else{
				features[curClass][j][s] = 1;
				//如果该特征取值从来没有出现过，增加M
				map<string, int>::iterator it2 = features[1 - curClass][j].find(s);
				if(it2 == features[1 - curClass][j].end())  M[j] ++;
			}
		}
	}
	outputLoadData();
	fclose(stdin);
}

//计算P(xi|y)，表示在@y类别的条件下，第@i个特征维度取值为@x的概率
double P(string x, int i, int y) {
	double result = 0.0;
	if(continuous[i]) {
		double num = atof(x.c_str());
		int index = (num - minValue[i]) / gap[i];
		stringstream ss;
	    ss << index;
	    x = "";
	    ss >> x;
	}
	result = ((double)features[y][i][x] + alpha) / ((double)N[y] + alpha * M[i]);
	return result;
}

bool classify(vector<string> &person) {
	double possibility[2];

	for(int ck = LOW50; ck <= HIGH50; ck ++) {
		double Py = (double)N[ck] / (double)totalTrain;
		possibility[ck] = log(Py);		
		for(int i = 0; i < FEATURE_NUM; i ++) {
			possibility[ck] = possibility[ck] + log(P(person[i], i, ck));
		}
	}
	int trueClass = (person[FEATURE_NUM] == "<=50K." ? LOW50 : HIGH50);
	all[trueClass] ++;
	if(possibility[trueClass] > possibility[1 - trueClass]) {
		correct[trueClass] ++;
		pred[trueClass] ++;
		return true;
	}
	pred[1 - trueClass] ++;
	return false;
}

int main() {
	init();
	loadData();
	ifstream fin("../data/adult.test");
	string s;
	cout << endl << endl;
	cout << "********TESTING**********" << endl;
	while(fin >> s) {
		totalTest ++;
		vector<string> person = split(s, ',');
		if(classify(person))  success ++;
	}
	cout << "Total Test Case: " << totalTest << endl;
	cout << "Accuracy: " << (double)success / (double)totalTest * 100 << "%" << endl;
	cout << "Precision: ≤50K " << (double)correct[LOW50] / (double) pred[LOW50] * 100 << "%";
	cout << "; >50K " << (double)correct[HIGH50] / (double) pred[HIGH50] * 100 << "%" << endl;
	cout << "Recall: ≤50K " << (double)correct[LOW50] / (double) all[LOW50] * 100 << "%";
	cout << "; >50K " << (double)correct[HIGH50] / (double) all[HIGH50] * 100 << "%" << endl;
	cout << "F1: ≤50K " << 1 / (0.5 * ((double)pred[LOW50] / (double) correct[LOW50] + (double)all[LOW50] / (double) correct[LOW50])) << endl;
	cout << "; >50K " << 1 / (0.5 * ((double)pred[HIGH50] / (double) correct[HIGH50] + (double)all[HIGH50] / (double) correct[HIGH50])) << endl;
	cout << "*******TEST OVER*********" << endl;
	fin.close();
	fclose(stdout);
	return 0;
}