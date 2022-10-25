#pragma once
#include <iostream>
#include <windows.h>  
using namespace std;


class serail
{
public:
	//设置参数
	bool setconfig(int baudrate);

	//打开
	bool open(LPCSTR COMx, int baudrate);

	//关闭
	bool close();

	//发送
	bool send(char* str);
	bool send(const char* str);

	//读取
	string read();

private:

}ser;

