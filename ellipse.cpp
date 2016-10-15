#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>
#include<vector>
//#include "Header.h"
//#include"Source1.cpp"
//int discriminantAnalysis(Mat_<uchar> src, Mat_<uchar>& dst);
using namespace cv;
#define THRESH_MAX 255


char trackbarNameThreshold1[] = "Threshold1";//トラックバーの名前

int discriminantAnalysis(Mat_<uchar> src, Mat_<uchar>& dst){
	/* ヒストグラム作成 */
	std::vector<int> hist(256, 0);  // 0-255の256段階のヒストグラム（要素数256、全て0で初期化）
	for (int y = 0; y < src.rows; ++y){
		for (int x = 0; x < src.cols; ++x){
			hist[static_cast<int>(src(y, x))]++;  // 輝度値を集計
		}
	}
	/* 判別分析法 */
	int t = 0;  // 閾値
	double max = 0.0;  // w1 * w2 * (m1 - m2)^2 の最大値

	for (int i = 0; i < 256; ++i){
		int w1 = 0;  // クラス１の画素数
		int w2 = 0;  // クラス２の画素数
		long sum1 = 0;  // クラス１の平均を出すための合計値
		long sum2 = 0;  // クラス２の平均を出すための合計値
		double m1 = 0.0;  // クラス１の平均
		double m2 = 0.0;  // クラス２の平均

		for (int j = 0; j <= i; ++j){
			w1 += hist[j];
			sum1 += j * hist[j];
		}

		for (int j = i + 1; j < 256; ++j){
			w2 += hist[j];
			sum2 += j * hist[j];
		}

		if (w1)
			m1 = (double)sum1 / w1;

		if (w2)
			m2 = (double)sum2 / w2;

		double tmp = ((double)w1 * w2 * (m1 - m2) * (m1 - m2));

		if (tmp > max){
			max = tmp;
			t = i;
		}
	}

	/* tの値を使って２値化 */
	for (int y = 0; y < src.rows; ++y){
		for (int x = 0; x < src.cols; ++x){
			if (src(y, x) < t)
				dst(y, x) = 0;
			else
				dst(y, x) = 255;
		}
	}

	return t;
}



int main(){
	//int a;
	Mat gray_img, bin_img, dst_img;
	Mat src_img = imread("../snap3.png", 1); //3チャンネルカラー画像
	Mat srchist_img = imread("../snap3.png", 0);//グレースケール画像
	if (!src_img.data)return-1;
	if (!srchist_img.data)return-1;
	
	resize(src_img, src_img, Size(), 0.3, 0.3);//元画像のリサイズ
	cvtColor(src_img, gray_img, CV_BGR2GRAY);//グレースケール変換

	//ヒストグラムを描画する画像割り当て
	const int ch_width = 400, ch_height = 300;
	Mat hist_img(Size(ch_width,ch_height),CV_8UC3,Scalar::all(255));

	Mat hist;
	const int hdims[] = { 256 };
	const float hranges[] = { 0, 256 };
	const float* ranges[] = { hranges };
	double max_val = .0;
	namedWindow("bin image", 1);
	int levels1 = 170;
	createTrackbar(trackbarNameThreshold1,"bin_image",&levels1,THRESH_MAX);

	//namedWindow("bin image",1);
	//シングルチャンネルのヒストグラム計算
	//画像、画像枚数、計算するチャンネル、マスク、ヒストグラム（出力）、
	//ヒストグラムの次元、ヒストグラムビンの下限
	calcHist(&srchist_img, 1, 0, Mat(), hist, 1, hdims, ranges);

	//最大値の計算
	minMaxLoc(hist, 0, &max_val);

	//ヒストグラムのスケーリングと描画
	Scalar color = Scalar::all(100);
	//スケーリング
	hist = hist * (max_val ? ch_height / max_val : 0.);
	for (int j = 0; j < hdims[0]; ++j){
		int bin_w = saturate_cast<int>((double)ch_width / hdims[0]);
		rectangle(hist_img,
			Point(j*bin_w, hist_img.rows),
			Point((j + 1)*bin_w,
			hist_img.rows - saturate_cast<int>(hist.at<float>(j))),
			color, -1);

	}
	/*
	std::vector<std::vector<Point>> contors;
	//画像の２値化
	threshold(gray_img, bin_img, 100, 255, THRESH_BINARY | THRESH_OTSU);
	//輪郭の検出
	findContours(bin_img, contors, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	for (int i = 0; i < contors.size(); ++i){
		size_t count = contors[i].size();
		if (count < 400) continue; //(小さすぎるor大きすぎる)輪郭を除外
		Mat pointsf;
		Mat(contors[i]).convertTo(pointsf, CV_32F);
		//楕円フィッティング
		RotatedRect box = fitEllipse(pointsf);
		//楕円の描画
		ellipse(src_img, box, Scalar(0, 0, 255), 2, CV_AA);

	}
	*/
	/*
	namedWindow("Image",CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
	namedWindow("Histogram",CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
	namedWindow("fit ellipse", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	//namedWindow("bin image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	imshow("fit ellipse", src_img);
	//imshow("bin image", bin_img);
	resize(srchist_img,srchist_img,Size(),0.3,0.3);
	imshow("Image",srchist_img);
	imshow("Histogram",hist_img);
	*/

	while (1){
		calcHist(&srchist_img, 1, 0, Mat(), hist, 1, hdims, ranges);

		//最大値の計算
		minMaxLoc(hist, 0, &max_val);

		//ヒストグラムのスケーリングと描画
		Scalar color = Scalar::all(100);
		//スケーリング
		hist = hist * (max_val ? ch_height / max_val : 0.);
		for (int j = 0; j < hdims[0]; ++j){
			int bin_w = saturate_cast<int>((double)ch_width / hdims[0]);
			rectangle(hist_img,
				Point(j*bin_w, hist_img.rows),
				Point((j + 1)*bin_w,
				hist_img.rows - saturate_cast<int>(hist.at<float>(j))),
				color, -1);

		}
		std::vector<std::vector<Point>> contors;
		threshold(gray_img, bin_img, levels1, 255, THRESH_BINARY);
		findContours(bin_img, contors, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		imshow("bin image", bin_img);
		for (int i = 0; i < contors.size(); ++i){
			size_t count = contors[i].size();
			if (count < 200) continue; //(小さすぎるor大きすぎる)輪郭を除外
			Mat pointsf;
			Mat(contors[i]).convertTo(pointsf, CV_32F);
			//楕円フィッティング
			RotatedRect box = fitEllipse(pointsf);
			//楕円の描画
			ellipse(src_img, box, Scalar(0, 0, 255), 2, CV_AA);

		}
		levels1++;
		std::cout << levels1 << std::endl;
		if (levels1 == 180) {
			//levels1 = 160;
			break;
		}
		//namedWindow("Image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		namedWindow("Histogram", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		namedWindow("fit ellipse", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		namedWindow("bin image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		imshow("fit ellipse", src_img);
		imshow("bin image", bin_img);
		//resize(srchist_img, srchist_img, Size(), 0.3, 0.3);
		//imshow("Image", srchist_img);
		imshow("Histogram", hist_img);
		waitKey(1000);
	}
}


