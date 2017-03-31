#include <iostream>
#include <map>
#include <vector>
#include <string.h>
#include <cmath>
#include <fstream>
#include <stdlib.h> 
#include <time.h> 
#include <iomanip>
#include "tools.h"

using namespace std;

#define FEATURE_NUM 14 //特征数量
#define LOW50 0
#define HIGH50 1
#define alpha 0.2  //Laplace平滑的系数
#define pi 3.1415926
#define K 4

double TRAIN_RATE = 1; //用于训练的样本比例

double N[2];  //两种类型人群的总人数，0表示≤50K，1表示>50K
double totalNum, totalTrain, totalTest;  //数据总数、训练集总数、测试集总数
vector< vector<string> > samples;  //训练集
vector< vector<string> > completeSamples;
map<string, int> features[2][FEATURE_NUM];  //表示某一类别下、某一个特征中该取值在当前类别中的数量，用于取离散值的特征
double u[2][FEATURE_NUM];  //表示某一类别下，某一特征取值的均值
double sita[2][FEATURE_NUM];  //表示某一类别下，某一特征取值的方差，这两个都用于连续取值的特征
bool continuous[FEATURE_NUM] = {1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0};
double M[FEATURE_NUM];  //表示某一维特征的最大取值个数（对离散取值而言），用于Laplace平滑
long success = 0;
long correct[2] = {0}, pred[2] = {0}, all[2] = {0};

bool check[40000];

vector<int> belongTo;
vector< vector<string> > clusters(K, vector<string>(FEATURE_NUM, ""));
double maxValue[FEATURE_NUM];


void init() {
	memset(N, 0, sizeof(N));
	totalNum = 0; totalTrain = 0; totalTest = 0;
	memset(u, 0, sizeof(u));
	memset(sita, 0, sizeof(sita));
	memset(M, 0, sizeof(M));
	memset(check, 0, sizeof(check));
	srand((unsigned)time(NULL)); 
	memset(maxValue, 0, sizeof(maxValue));
}

//计算P(xi|y)，表示在@y类别的条件下，第@i个特征维度取值为@x的概率
double P(string x, int i, int y) {
	double result = 0.0;
	if(continuous[i]) {
		double num = atof(x.c_str());
		num = num / maxValue[i];
		result = (1 / sqrt(2.0 * pi * sita[y][i])) * exp(-((num - u[y][i]) * (num - u[y][i]) / (2.0 * sita[y][i])));
	}
	else {
		result = ((double)features[y][i][x] + alpha) / ((double)N[y] + alpha * M[i]);
	}
	return result;
}

double diff(vector<string> &person1, vector<string> &person2) {
	double contDiff = 0, disperseDiff = 0;
	double disperse = 0;
	for(int i = 0; i < FEATURE_NUM; i ++) {
		if(person1[i] == "?" || person2[i] == "?")  continue;
		if(continuous[i]) {
			double num1 = atof(person1[i].c_str()) / maxValue[i];
			double num2 = atof(person2[i].c_str()) / maxValue[i];
			contDiff += (num1 - num2) * (num1 - num2);
		}
		else {
			disperse ++;
			if(person1[i] != person2[i])  disperseDiff ++;
		}
	}
	contDiff = sqrt(contDiff);
	disperseDiff /= disperse;
	return contDiff + disperseDiff;
}

int classify_cluster(vector<string> &person) {
	int result;
	double minDiff = INT_MAX;
	for(int i = 0; i < K; i ++) {
		double diff_i = diff(clusters[i], person);
		if(diff_i < minDiff) {
			minDiff = diff_i;
			result = i;
		}
	}
	return result;
}

void K_means_init() {
	cout << endl << endl;
	cout << "********K-means Working**********" << endl;
	int completeSize = completeSamples.size();
	for(int i = 0; i < K; i ++) {
		clusters[i] = completeSamples[rand() % completeSize];
	}
	bool flag = true;
	int iterateCount = 0;
	while(flag) {
		iterateCount ++;
		flag = false;
		int count[K];  double sum[K][FEATURE_NUM];
		memset(count, 0, sizeof(count));
		memset(sum, 0, sizeof(sum));
		//重新为每个人归类
		for(int i = 0; i < completeSize; i ++) {
			int classify_result = classify_cluster(completeSamples[i]);
			if(classify_result != belongTo[i]) {
				flag = true;
				belongTo[i] = classify_result;
			}
			count[classify_result] ++;
		}
		//记录各个类别中各个特征的信息
		map<string, int> maxTime[K][FEATURE_NUM];
		for(int i = 0; i < completeSize; i ++) {
			for(int j = 0; j < FEATURE_NUM; j ++) {
				int cluster = belongTo[i];
				vector<string> person = completeSamples[i];
				if(continuous[j]) {
					double num = atof(person[j].c_str());
					sum[cluster][j] += num;
				}
				else {
					if(maxTime[cluster][j].find(person[j]) != maxTime[cluster][j].end()) {
						maxTime[cluster][j][person[j]] ++;
					}
					else {
						maxTime[cluster][j][person[j]] = 0;
					}
				}
			}
		}
		//重新计算所有簇的中心
		for(int i = 0; i < K; i ++) {
			for(int j = 0; j < FEATURE_NUM; j ++) {
				if(continuous[j]) {
					clusters[i][j] = to_string(sum[i][j] / count[i]);
				}
				else {
					map<string, int>::iterator it;
					int Max = 0;
					for(it = maxTime[i][j].begin(); it != maxTime[i][j].end(); it ++) {
						if(it -> second > Max) {
							Max = it -> second;
							clusters[i][j] = it -> first;
						}
					}
				}
			}
		}
		memset(maxTime, 0, sizeof(maxTime));
	}
	cout << "Iterate Times: " << iterateCount << endl;
	cout << "********K-means Done**********" << endl << endl << endl; 
}

void outputLoadData() {
	cout << "TRAIN_RATE: " << TRAIN_RATE << endl;
	cout << "Traning Scale: " << totalTrain << endl;
	cout << "≤50K: " << N[LOW50] << " >50K: " << N[HIGH50] << endl;
	cout << endl << "u: " << endl;
	for(int i = 0; i < 2; i ++) {
		for(int j = 0; j < FEATURE_NUM; j ++) {
			if(!continuous[j])  cout << 0 << " ";
			else cout << u[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl << "sita: " << endl;
	for(int i = 0; i < 2; i ++) {
		for(int j = 0; j < FEATURE_NUM; j ++) {
			if(!continuous[j])  cout << 0 << " ";
			else cout << sita[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl << "M: " << endl;
	for(int i = 0; i < FEATURE_NUM; i ++) {
		if(continuous[i])  cout << "--" << " ";
		else cout << M[i] << " ";
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
		int i;
		for(i = 0; i < FEATURE_NUM; i ++) {
			if(person[i] == "?") 
				break;
		}
		if(i == FEATURE_NUM) {
			completeSamples.push_back(person);
			belongTo.push_back(0);
		}
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
			//如果该特征缺失，在训练的时候就不加入计算
			if(person[i] == "?")  continue;
			//连续取值特征第一次遍历只算均值
			if(continuous[i]) {
				double num = atof(person[i].c_str());
				if(num > maxValue[i])  maxValue[i] = num;
			}
			else {
				map<string, int>::iterator it = features[curClass][i].find(person[i]);
				//如果已经记录过该特征
				if(it != features[curClass][i].end()) {
					features[curClass][i][person[i]] ++;
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
	//计算均值
	for(int i = 0; i < totalTrain; i ++) {
		int curClass = (samples[i][FEATURE_NUM] == "<=50K." ? LOW50 : HIGH50);
		for(int j = 0; j < FEATURE_NUM; j ++) {
			if(!continuous[j])  continue;
			double num = atof(samples[i][j].c_str());
			num = num / maxValue[j];
			u[curClass][j] += num;
		}
	}
	for(int i = 0; i < FEATURE_NUM; i ++) {
		if(!continuous[i])  continue;
		u[LOW50][i] /= N[LOW50];
		u[HIGH50][i] /= N[HIGH50];
	}
	//计算方差
	for(int i = 0; i < totalTrain; i ++) {
		int curClass = (samples[i][FEATURE_NUM] == "<=50K." ? LOW50 : HIGH50);
		for(int j = 0; j < FEATURE_NUM; j ++) {
			if(!continuous[j])  continue;
			double num = atof(samples[i][j].c_str());
			num = num / maxValue[j];
			if(samples[i][j] != "?")  sita[curClass][j] += (u[curClass][j] - num) * (u[curClass][j] - num);
		}
	}
	for(int i = 0; i < FEATURE_NUM; i ++) {
		if(!continuous[i])  continue;
		sita[LOW50][i] /= N[LOW50];
		sita[HIGH50][i] /= N[HIGH50];
	}
	outputLoadData();
	fclose(stdin);
}

bool classify(vector<string> &person) {
	double possibility[2];

	for(int ck = LOW50; ck <= HIGH50; ck ++) {
		double Py = (double)N[ck] / (double)totalTrain;
		possibility[ck] = log(Py);		
		for(int i = 0; i < FEATURE_NUM; i ++) {
			string s = person[i];
			if(person[i] == "?") {
				int cluster = classify_cluster(person);
				s = clusters[cluster][i];
			}
			possibility[ck] = possibility[ck] + log(P(s, i, ck));
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
	K_means_init();
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