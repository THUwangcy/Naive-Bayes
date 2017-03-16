#include <iostream>
#include <map>
#include <vector>
#include <string.h>

using namespace std;

//分隔字符串
vector<string> split(const string &s, const char seperator){
	vector<string> result;
	typedef string::size_type string_size;
	string_size i = 0, j = 0;
	int length = s.size();

	for(i = 0; i < length; i ++) {
		j = i;
		while(j < length && s[j] != seperator) j ++;
		result.push_back(s.substr(i, j - i));
		i = j;
	}

	return result;
}