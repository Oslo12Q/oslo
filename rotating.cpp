// ImageHandle_ExaminationReport.cpp : 定义控制台应用程序的入口点。
//

#include <iostream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <ctime>
#include <math.h>


#define ERROR 1234
static int flag = 0;

using namespace std;
using namespace cv;

//度数转换
double DegreeTrans(double theta)
{
	//cout << "DegreeTrans!" << endl;
    double res = theta / CV_PI * 180;
    return res;
}


//逆时针旋转图像degree角度（原尺寸）    
void rotateImage(Mat src, Mat& img_rotate, double degree)
{
	//cout << "rotateImage!" << endl;
    //旋转中心为图像中心    
    Point2f center;
    center.x = float(src.cols / 2.0);
    center.y = float(src.rows / 2.0);
    int length = 0;
    length = sqrt(src.cols*src.cols + src.rows*src.rows);
    //计算二维旋转的仿射变换矩阵  
    Mat M = getRotationMatrix2D(center, degree, 1);
    warpAffine(src, img_rotate, M, Size(length, length), 1, 0, Scalar(255,255,255));//仿射变换，背景色填充为白色  
}
float find_max_seq(vector<float> a)
{
	//cout << "find_max_seq!" << endl;
	float ele=0.0;
    int count = 1, count1 = 1;
	for(int i = 0; i < a.size()-1; i++) {
        while(i < a.size()-1 && a[i]>a[i+1]-0.01) {
            count++;
		
            i++;
        }
        if(count >= count1) {
            count1 = count;
			ele = a[i];
        }
        count = 1;
    }

    return ele;
}
//通过霍夫变换计算角度
double CalcDegree(const Mat &srcImage, Mat &dst)
{
	//cout << "CalcDegree!" << endl;
    Mat midImage, dstImage;
    Canny(srcImage, midImage, 50, 200, 3);
    cvtColor(midImage, dstImage, CV_GRAY2BGR);

    //通过霍夫变换检测直线
    vector<Vec2f> lines;
    HoughLines(midImage, lines, 1, CV_PI / 180, 300, 0, 0);//第5个参数就是阈值，阈值越大，检测精度越高


    //由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
    //所以根据阈值由大到小设置了三个阈值，如果经过大量试验后，可以固定一个适合的阈值。

    if (!lines.size())
    {
        HoughLines(midImage, lines, 1, CV_PI / 180, 200, 0, 0);
    }
  

    if (!lines.size())
    {
        HoughLines(midImage, lines, 1, CV_PI / 180, 150, 0, 0);
    }

    if (!lines.size())
    {
        return ERROR;
    }
	vector<float> arph_;
    //依次画出每条线段
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0];
        float theta = lines[i][1];
        Point pt1, pt2;

        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        //只选角度最小的作为旋转角度
		arph_.push_back(theta);

        line(dstImage, pt1, pt2, Scalar(55, 100, 195), 1); //Scalar函数用于调节线段颜色, LINE_AA

    }
	sort(arph_.begin(), arph_.end());


   float average = find_max_seq(arph_);//sum / lines.size(); //对所有角度求平均，这样做旋转效果会更好
   

    double angle =DegreeTrans(average) - 90;

    rotateImage(dstImage, dst, angle);
    //imshow("直线探测效果图2", dstImage);
    return angle;
}


Mat ImageRecify(const char* pInFileName, const char* pOutFileName)
{
	//cout << "imageRecify!" << endl;
    double degree;
    Mat src = imread(pInFileName);

    Mat dst;
    //倾斜角度矫正
    degree = CalcDegree(src,dst);
    if (degree == ERROR)
    {
        ;
    }
    rotateImage(src, dst, degree);
   // imshow("旋转调整后", dst);
	//waitKey();

	imwrite(pOutFileName,dst);
	return dst;
}


int main(int argc,char *argv[]){	
	
	Mat b = ImageRecify(argv[1],argv[2]);
	return 0;
}
