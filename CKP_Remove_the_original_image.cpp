#include<iostream>
#include"stdio.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int flag;
void chao_thinimage(Mat &srcimage)//单通道、二值化后的图像
{
	vector<Point> deletelist1;
	int Zhangmude[9];
	int nl = srcimage.rows;
	int nc = srcimage.cols;
	while (true)
	{
		for (int j = 1; j < (nl - 1); j++)
		{
			uchar* data_last = srcimage.ptr<uchar>(j - 1);
			uchar* data = srcimage.ptr<uchar>(j);
			uchar* data_next = srcimage.ptr<uchar>(j + 1);
			for (int i = 1; i < (nc - 1); i++)
			{
				if (data[i] == 255)
				{
					Zhangmude[0] = 1;
					if (data_last[i] == 255) Zhangmude[1] = 1;
					else  Zhangmude[1] = 0;
					if (data_last[i + 1] == 255) Zhangmude[2] = 1;
					else  Zhangmude[2] = 0;
					if (data[i + 1] == 255) Zhangmude[3] = 1;
					else  Zhangmude[3] = 0;
					if (data_next[i + 1] == 255) Zhangmude[4] = 1;
					else  Zhangmude[4] = 0;
					if (data_next[i] == 255) Zhangmude[5] = 1;
					else  Zhangmude[5] = 0;
					if (data_next[i - 1] == 255) Zhangmude[6] = 1;
					else  Zhangmude[6] = 0;
					if (data[i - 1] == 255) Zhangmude[7] = 1;
					else  Zhangmude[7] = 0;
					if (data_last[i - 1] == 255) Zhangmude[8] = 1;
					else  Zhangmude[8] = 0;
					int whitepointtotal = 0;
					for (int k = 1; k < 9; k++)
					{
						whitepointtotal = whitepointtotal + Zhangmude[k];
					}
					if ((whitepointtotal >= 2) && (whitepointtotal <= 6))
					{
						int ap = 0;
						if ((Zhangmude[1] == 0) && (Zhangmude[2] == 1)) ap++;
						if ((Zhangmude[2] == 0) && (Zhangmude[3] == 1)) ap++;
						if ((Zhangmude[3] == 0) && (Zhangmude[4] == 1)) ap++;
						if ((Zhangmude[4] == 0) && (Zhangmude[5] == 1)) ap++;
						if ((Zhangmude[5] == 0) && (Zhangmude[6] == 1)) ap++;
						if ((Zhangmude[6] == 0) && (Zhangmude[7] == 1)) ap++;
						if ((Zhangmude[7] == 0) && (Zhangmude[8] == 1)) ap++;
						if ((Zhangmude[8] == 0) && (Zhangmude[1] == 1)) ap++;
						if (ap == 1)
						{
							if ((Zhangmude[1] * Zhangmude[7] * Zhangmude[5] == 0) && (Zhangmude[3] * Zhangmude[5] * Zhangmude[7] == 0))
							{
								deletelist1.push_back(Point(i, j));
							}
						}
					}
				}
			}
		}
		if (deletelist1.size() == 0) break;
		for (size_t i = 0; i < deletelist1.size(); i++)
		{
			Point tem;
			tem = deletelist1[i];
			uchar* data = srcimage.ptr<uchar>(tem.y);
			data[tem.x] = 0;
		}
		deletelist1.clear();

		for (int j = 1; j < (nl - 1); j++)
		{
			uchar* data_last = srcimage.ptr<uchar>(j - 1);
			uchar* data = srcimage.ptr<uchar>(j);
			uchar* data_next = srcimage.ptr<uchar>(j + 1);
			for (int i = 1; i < (nc - 1); i++)
			{
				if (data[i] == 255)
				{
					Zhangmude[0] = 1;
					if (data_last[i] == 255) Zhangmude[1] = 1;
					else  Zhangmude[1] = 0;
					if (data_last[i + 1] == 255) Zhangmude[2] = 1;
					else  Zhangmude[2] = 0;
					if (data[i + 1] == 255) Zhangmude[3] = 1;
					else  Zhangmude[3] = 0;
					if (data_next[i + 1] == 255) Zhangmude[4] = 1;
					else  Zhangmude[4] = 0;
					if (data_next[i] == 255) Zhangmude[5] = 1;
					else  Zhangmude[5] = 0;
					if (data_next[i - 1] == 255) Zhangmude[6] = 1;
					else  Zhangmude[6] = 0;
					if (data[i - 1] == 255) Zhangmude[7] = 1;
					else  Zhangmude[7] = 0;
					if (data_last[i - 1] == 255) Zhangmude[8] = 1;
					else  Zhangmude[8] = 0;
					int whitepointtotal = 0;
					for (int k = 1; k < 9; k++)
					{
						whitepointtotal = whitepointtotal + Zhangmude[k];
					}
					if ((whitepointtotal >= 2) && (whitepointtotal <= 6))
					{
						int ap = 0;
						if ((Zhangmude[1] == 0) && (Zhangmude[2] == 1)) ap++;
						if ((Zhangmude[2] == 0) && (Zhangmude[3] == 1)) ap++;
						if ((Zhangmude[3] == 0) && (Zhangmude[4] == 1)) ap++;
						if ((Zhangmude[4] == 0) && (Zhangmude[5] == 1)) ap++;
						if ((Zhangmude[5] == 0) && (Zhangmude[6] == 1)) ap++;
						if ((Zhangmude[6] == 0) && (Zhangmude[7] == 1)) ap++;
						if ((Zhangmude[7] == 0) && (Zhangmude[8] == 1)) ap++;
						if ((Zhangmude[8] == 0) && (Zhangmude[1] == 1)) ap++;
						if (ap == 1)
						{
							if ((Zhangmude[1] * Zhangmude[3] * Zhangmude[5] == 0) && (Zhangmude[3] * Zhangmude[1] * Zhangmude[7] == 0))
							{
								deletelist1.push_back(Point(i, j));
							}
						}
					}
				}
			}
		}
		if (deletelist1.size() == 0) break;
		for (size_t i = 0; i < deletelist1.size(); i++)
		{
			Point tem;
			tem = deletelist1[i];
			uchar* data = srcimage.ptr<uchar>(tem.y);
			data[tem.x] = 0;
		}
		deletelist1.clear();
	}
}

struct Pic{
	bool pic_class;
	int y1, y2, y3;
	int x_vertical;
};

Pic findpoint_horizontal(Mat horizontal) {
	vector<int> endtemp;

	//寻找左上角点
	for (int i = 200; i < horizontal.rows - 100; i++)
	{
		uchar *data = horizontal.ptr<uchar>(i);
		for (int j = 0; j < horizontal.cols; j++)
		{
			if (data[j] == 255 && data[j - 1] == 0)
			{
				if (endtemp.size() == 0 || i - endtemp[endtemp.size() - 1] > 20 ) {
					endtemp.push_back(i);
				}
			}
		}
	}
	
	Pic pic;
	if (endtemp.size() == 3) {
		pic.y1 = endtemp[0];
		pic.y2 = endtemp[1];
		pic.y3 = endtemp[2];
	}
	else if (endtemp.size() < 3) {
		pic.y1 = -1;
		pic.y2 = -1;
		pic.y3 = -1;
	}
	else if (endtemp.size() > 3) {
		pic.y1 = endtemp[0];
		pic.y2 = endtemp[1];
		for (int i = 0; i < endtemp.size(); i++) {
			if (endtemp[i] > 1000) {
				pic.y3 = endtemp[2];
			}
		}

	}
	return pic;
}

int findpoint_vertical(Mat vertical) {
	vector<int> temp;
	//寻找左上角点
	for (int i = 200; i < vertical.rows - 100; i++)
	{
		uchar *data = vertical.ptr<uchar>(i);
		for (int j = 200; j < vertical.cols-100; j++)
		{
			if (data[j] == 255 && data[j - 1] == 0)
			{
				if (temp.size() == 0 || j - temp[temp.size() - 1] > 10) {
					temp.push_back(j);
				}
			}
		}
	}
	if (temp.size() == 0) {
		return 0;
	}
	if (temp.size() == 1) {
		return temp[0];
	}
	if (temp.size() > 1) {
		for (int i = 0; i < temp.size(); i++) {
			//cout << temp[i] << "  ";
			if (temp[i] > 1000)
				return temp[i];
		}
		return -1;
	}
}

Pic contour(Mat srcImg) {
	Mat dstImg;
	cvtColor(srcImg, dstImg, CV_BGR2GRAY);
	Mat bw;
	adaptiveThreshold(~dstImg, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
//	imwrite("bw.jpg", bw);

	Mat horizontal = bw.clone();
	Mat vertical = bw.clone();

	int scale = 20; //这个值越大，检测到的直线越多
	int horizontalsize = horizontal.cols / scale;

	// 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
	Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));

	// 先腐蚀再膨胀
	erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	//imshow("horizontal", horizontal);

	int verticalsize = vertical.rows / scale;
	Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
	erode(vertical, vertical, verticalStructure, Point(-1, -1));
	dilate(vertical, vertical, verticalStructure, Point(-1, -1));
	//imshow("vertical", vertical);

	//Mat mask = horizontal + vertical;
	//imwrite("mask.jpg", mask);

	chao_thinimage(horizontal);
	chao_thinimage(vertical);

	Pic pic;
	pic = findpoint_horizontal(horizontal);
	if (pic.y1 = -1) return pic;
	pic.x_vertical = findpoint_vertical(vertical);
	if (pic.x_vertical == -1) pic.x_vertical = 0;
	if (pic.y2 - pic.y1 > 100) {
		pic.pic_class = false;
	}
	else {
		pic.pic_class = true;
	}
	return pic;
}

void findLine(IplImage* raw, IplImage* dst) {
	IplImage* src = cvCloneImage(raw);  // clone the input image  
	IplImage* canny = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);  // create a tmp image head to save gradient image  
	cvCanny(src, canny, 20, 200, 3);  // Generate its gradient image  
	CvMemStorage* stor = cvCreateMemStorage(0);
	CvSeq* lines = NULL;
	// find a line whose length bigger than 200 pixels  
	lines = cvHoughLines2(canny, stor, CV_HOUGH_PROBABILISTIC, 1, CV_PI / 180, 80, 200, 30);
	cvZero(dst);
	CvPoint maxStart = cvPoint(0, 0), maxEnd = cvPoint(0, 0);   // save the coordinate of the head and rear of the line we want  
	int maxDistance = 0;  // The maximum distance of all lines found by [cvHoughLines2]  
	for (int i = 0; i < lines->total; i++) {  // lines->total: the number of lines   
		// variable 'lines' is a sequence, [cvGetSeqElem] gets the (i)th line, and it returns its head and rear.  
		CvPoint* line = (CvPoint*)cvGetSeqElem(lines, i);
		// line[0] and line[1] is respectively the line's coordinate of its head and rear  
		if (abs(line[0].x - line[1].x) > maxDistance) {
			/*  It's a trick because the line is almost horizontal.
			 strictly, it should be
			 sqrt((line[0].x - line[1].x)*(line[0].x - line[1].x)+(line[0].y - line[1].y)*(line[0].x - line[1].x))
			*/
			maxDistance = abs(line[0].x - line[1].x);
			maxStart = line[0];
			maxEnd = line[1];
		}
	}
	cvLine(dst, maxStart, maxEnd, cvScalar(255), 1);    // draw the white line[cvScalar(255)] in a black background  
	//2018.09.19发现个别图片会报错 原因:maxend没有初始化
	cvReleaseImage(&src);                               // free the memory
	cvReleaseMemStorage(&stor);
}

void erase(IplImage* raw) {
	IplImage* src = cvCloneImage(raw);
	/*Binarization and inverse the black and white because the function next only find white area while
	the word in image is black.*/
	cvThreshold(src, src, 120, 255, CV_THRESH_BINARY_INV);
	// create some space to save the white areas but we access it via variable 'cont'
	CvMemStorage* stor = cvCreateMemStorage(0);
	CvSeq* cont;
	cvFindContours(src, stor, &cont, sizeof(CvContour), CV_RETR_EXTERNAL); // find the white regions  
	for (; cont; cont = cont->h_next) { // Traversal
		if (fabs(cvContourArea(cont)) < 3)  // if its Area smaller than 15, we fill it with white[cvScalar(255)]  
			cvDrawContours(raw, cont, cvScalar(255), cvScalar(255), 0, CV_FILLED, 8);
	}
	cvReleaseImage(&src);
}

Mat DeleteLine(Mat input_image) {
	if (input_image.empty()) {
		return input_image;
	}

	IplImage iplimage = input_image;
	IplImage* src = &iplimage, *gray;
	IplImage* dst = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	IplImage* binary = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	gray = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);


	if (src->nChannels == 1)
		gray = cvCloneImage(src);
	else if (src->nChannels == 3)
		cvCvtColor(src, gray, CV_RGB2GRAY);
	else
		return input_image;

	cvThreshold(gray, binary, 120, 255, CV_THRESH_OTSU);
	findLine(gray, dst);

	for (int row = 0; row < binary->height; row++)
		for (int col = 0; col < binary->width; col++) {
			if (cvGet2D(dst, row, col).val[0] == 255) {
				int up = 0, down = 0;
				int white = 0;
				for (int i = row; i >= 0; i--) {
					if (cvGet2D(binary, i, col).val[0] == 0) {
						up++;
						white = 0;
					}
					else white++;
					if (white > 2)
						break;
				}
				white = 0;
				for (int i = row; i < binary->height; i++) {
					if (cvGet2D(binary, i, col).val[0] == 0) {
						down++;
						white = 0;
					}
					else white++;
					if (white > 2)
						break;
				}
				if (up + down < 8) {
					for (int i = -up; i <= down; i++) {
						if (row + i >= 0 && row + i < binary->height)
							cvSet2D(binary, row + i, col, cvScalar(255));
					}
				}
			}
		}
	erase(binary);
	input_image = cvarrToMat(binary);
	cvReleaseImage(&gray);
	cvReleaseImage(&dst);
	return input_image;
}

Mat Delete(Mat src_img1, Mat img2) {//清除直线
	Mat img1 = src_img1.clone();
	int rows = img1.rows;
	int cols = img1.cols;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			uchar t;
			if (img1.channels() == 1 && img2.channels() == 1) {
				t = img2.at<uchar>(i, j);
				if (t == 0)
					img1.at<uchar>(i, j) = 255;
			}
			else if (img1.channels() == 3 && img2.channels() == 3) {
				for (int k = 0; k < 3; k++) {
					t = img2.at<Vec3b>(i, j)[k];
					if (t == 0)
						img1.at<Vec3b>(i, j)[k] = 255;
				}
			}
			else {
				return src_img1;
			}
		}
	}
	return img1;
}

Mat Preprocessings(Mat b) {
	Mat dst1, dst2, result, binImg, gray_src, hline, vline;

	Mat src = DeleteLine(b);

	//imwrite("src.jpg",src);
	if (src.empty()) {
		return src;
	}
	char INPUT_WIN[] = "input image";
	char OUTPUT_WIN[] = "result image";

	// 变成灰度图像
	gray_src = src.clone();
	//imwrite("gray_src.jpg", gray_src);
	//变成二值图像
	adaptiveThreshold(~gray_src, binImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	//imwrite("binImg.jpg", binImg);

	//水平结构元素
	hline = getStructuringElement(MORPH_RECT, Size(src.cols / 16, 1), Point(-1, -1));
	if (src.rows / 16 > 0) {
		//垂直结构元素
		vline = getStructuringElement(MORPH_RECT, Size(1, src.rows / 16), Point(-1, -1));
	}
	else {
		vline = getStructuringElement(MORPH_RECT, Size(1, 1), Point(-1, -1));
	}

	// 形态学开操作——先腐蚀后膨胀
	morphologyEx(binImg, dst1, CV_MOP_OPEN, hline);
	morphologyEx(binImg, dst2, CV_MOP_OPEN, vline);
	bitwise_not(dst1, dst1);
	bitwise_not(dst2, dst2);

	result = Delete(Delete(src, dst1), dst2);

	result = DeleteLine(result);

	return result;
}

float getRatio(Mat input_image, int flag) {//获得图片像素黑色白色比
	if (input_image.empty())
		return -1;

	Mat src = input_image(Range(input_image.rows*0.05, input_image.rows*0.95), Range(input_image.cols*0.05, input_image.cols*0.95));

	int rows = src.rows;
	int cols = src.cols;
	int black = 0, white = 0;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			uchar t;
			if (src.channels() == 1) {
				if (src.at<uchar>(i, j) == 0)
					black++;
				else
					white++;
			}
			else
				;
		}
	}

	if (flag)
		return (float)black / white;
	else
		return (float)white / black;
}

int horizontalCut(Mat input_src, const char *out_cut, Mat pre_image, string dirname, Point points) {//切原图 含宽度判断 加矫正切分
	if (pre_image.empty()) {

		return -1;
	}

	Mat result;//输出图片
	Mat dst1 = input_src.clone();//imread(out);//imread(file);//原图

	Mat dst2 = pre_image.clone();//处理后的图
	//imshow("2",dst2);
	//waitKey();

	if (getRatio(pre_image, 1) < 0.001) {//过度处理
		;// imwrite("dst.jpg", dst1);
		return -1;
	}

	IplImage iplimage = pre_image;

	IplImage* src = &iplimage;

	IplImage *dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	cvThreshold(src, dst, 100, 255, CV_THRESH_BINARY);
	int* h = new int[dst->height];
	memset(h, 0, dst->height * 4);
	int x, y;
	CvScalar s, t;//可以用来存放4个double数值的数组，一般用来存放像素值（不一定是灰度值哦）的，最多可以存放4个通道的 
	for (y = 0; y < dst->height; y++) {
		for (x = 0; x < dst->width; x++) {
			s = cvGet2D(dst, y, x);

			if (s.val[0] == 0)
				h[y]++;
		}
	}


	vector<Mat> roiList;//用于储存分割出来的每个字符
	int startIndex = 0;//记录进入字符区的索引
	int endIndex = 0;//记录进入空白区域的索引
	bool inBlock = false;//是否遍历到了字符区内
	char szName[30] = { 0 };
	int width = 0;

	for (int i = 0; i < dst->height; ++i) {
		if (!inBlock && h[i] != 0) {//进入字符区了
			inBlock = true;
			startIndex = i;
		}
		else if (h[i] < 15 && inBlock || i == dst->height &&inBlock) {//进入空白区了

			//2018.11.23漏洞:手拍的照片可能底部有阴影 处理过后无法切出图片


			endIndex = i;
			inBlock = false;
			Mat roiImg, roiImg_src;
			if (endIndex - startIndex > 20) {//初始值为3 2018.08.13 17:30改为20 去除无效切割区域
				roiImg_src = dst1(Range(startIndex, endIndex + 1), Range(0, dst1.cols));
				roiImg = dst2(Range(startIndex, endIndex + 1), Range(0, dst1.cols));
				roiList.push_back(roiImg_src);

				if (getRatio(roiImg, 0) < 200) {//2018.09.05 13:17
					width += endIndex - startIndex;
					double wid = (double)(endIndex - startIndex) / (width / (flag + 1)) / 1.2;
					int int_wid = -1;

					if (wid < 2 && wid > 0)
						int_wid = 1;
					else
						int_wid = 0;


					char *img_name = new char[strlen(out_cut) + sizeof(flag) + strlen("-(,,,,)") + strlen(dirname.c_str()) + sizeof(0 + points.x) + sizeof(startIndex + points.y) + sizeof(endIndex - startIndex) + sizeof(dst1.cols) + sizeof(int_wid)];
					sprintf(img_name, "%s%d%s%d%s%d%s%d%s%d%s%d%s%s", out_cut, flag, "(", 0 + points.x, ",", startIndex + points.y, ",", dst1.cols, ",", endIndex - startIndex, ",", int_wid, ")-", dirname.c_str());
					imwrite(img_name, roiImg_src);
					flag++;

				}
			}
		}
	}

	//	cvWaitKey(0);
	cvDestroyAllWindows();
	cvReleaseImage(&dst);
	return 0;
}

int cut(Mat srcImg, Mat pre, Pic pic, const char* argv2, string name) {
	flag = 0;
	Point points;
	if (pic.y1 != -1) {
		Mat src_top = srcImg(Range(0, pic.y1), Range(0, srcImg.cols));
		Mat _src_top = pre(Range(0, pic.y1), Range(0, srcImg.cols));

		points.x = 0;
		points.y = pic.y1;
		horizontalCut(src_top, argv2, _src_top, name, points);
		//imwrite("src_top.jpg",src_top);

		Mat src_top1 = srcImg(Range(pic.y1, pic.y2), Range(0, srcImg.cols));
		int int_wid = 0;
		char *img_name = new char[strlen(argv2) + sizeof(flag) + strlen("-(,,,,)") + strlen(name.c_str()) + sizeof(0) + sizeof(pic.y1) + sizeof(pic.y2 - pic.y1) + sizeof(src_top1.cols) + sizeof(int_wid)];
		sprintf(img_name, "%s%d%s%d%s%d%s%d%s%d%s%d%s%s", argv2, flag, "(", 0, ",", pic.y1, ",", src_top1.cols, ",", pic.y2 - pic.y1, ",", int_wid, ")-", name.c_str());
		flag++;

		imwrite(img_name, src_top1);

		//imwrite("src_top1.jpg", src_top1);

		if (pic.x_vertical == 0) {
			Mat src_middle = srcImg(Range(pic.y2, pic.y3), Range(0, srcImg.cols));
			Mat _src_middle = pre(Range(pic.y2, pic.y3), Range(0, srcImg.cols));
			points.x = 0;
			points.y = pic.y2;
			horizontalCut(src_middle, argv2, _src_middle, name, points);
			//imwrite("src_middle.jpg", src_middle);
		}
		else {
			Mat src_middle_left = srcImg(Range(pic.y2, pic.y3), Range(0, pic.x_vertical));
			Mat _src_middle_left = pre(Range(pic.y2, pic.y3), Range(0, pic.x_vertical));
			points.x = 0;
			points.y = pic.y2;
			horizontalCut(src_middle_left, argv2, _src_middle_left, name, points);
			//imwrite("src_middle_left.jpg", src_middle_left);

			Mat src_middle_right = srcImg(Range(pic.y2, pic.y3), Range(pic.x_vertical, srcImg.cols));
			Mat _src_middle_right = pre(Range(pic.y2, pic.y3), Range(pic.x_vertical, srcImg.cols));
			points.x = pic.x_vertical;
			points.y = pic.y2;
			horizontalCut(src_middle_right, argv2, _src_middle_right, name, points);
			//imwrite("src_middle_right.jpg", src_middle_right);
		}
		Mat src_bottom = srcImg(Range(pic.y3, srcImg.rows), Range(0, srcImg.cols));
		Mat _src_bottom = pre(Range(pic.y3, srcImg.rows), Range(0, srcImg.cols));
		points.x = 0;
		points.y = pic.y3;
		horizontalCut(src_bottom, argv2, _src_bottom, name, points);
		//imwrite("src_bottom.jpg", src_bottom);
	}
	else {
		points.x = 0;
		points.y = 0;
		horizontalCut(srcImg, argv2, pre, name, points);
	}
	return 0;


}

int main(int argc,char *argv[])
{
	//char* argv[3];
	//argv[1] = "E:\\working\\ckp2.0\\ckp2.0\\test\\handle-img20181225_15040359.jpg";
	int i = 0;
	string name1 = argv[1];
	for (i = strlen(argv[1]) - 1; i >= 0; i--) {
		if (argv[1][i] == '/') {
			break;
		}
	}
	string name(name1.substr(i + 1, strlen(argv[1]) - 1));


	//argv[2] = "E:\\working\\ckp2.0\\ckp2.0\\test\\";
	
	Mat srcImg = imread(argv[1]);
	Mat pre = Preprocessings(srcImg);
	Pic pic = contour(srcImg);
	cut(srcImg, pre, pic, argv[2],name);
	//system("pause");
	return 0;
}
