#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>
#include<vector>

using namespace cv;
#define THRESH_MAX 255


char trackbarNameThreshold1[] = "Threshold1";//トラックバーの名前

int main(){
	Mat gray_img, bin_img, dst_img;

	Mat src_img = imread("../snap3.png", 1); //3チャンネルカラー画像
	Mat srchist_img = imread("../snap3.png", 0);//グレースケール画像
	if (!src_img.data)return-1;
	if (!srchist_img.data)return-1;


	resize(src_img, src_img, Size(), 0.3, 0.3);//元画像のリサイズ
	GaussianBlur(src_img, src_img, Size(5, 5), 2, 2);
	cvtColor(src_img, gray_img, CV_BGR2GRAY);//グレースケール変換

	//ヒストグラムを描画する画像割り当て
	const int ch_width = 400, ch_height = 300;
	Mat hist_img(Size(ch_width, ch_height), CV_8UC3, Scalar::all(255));

	Mat hist;
	const int hdims[] = { 256 };
	const float hranges[] = { 0, 256 };
	const float* ranges[] = { hranges };
	double max_val = .0;
	namedWindow("bin_image", 1);

	int levels1 = 135; //２値化閾値の設定
	createTrackbar(trackbarNameThreshold1, "bin_image", &levels1, THRESH_MAX);

	//シングルチャンネルのヒストグラム計算
	//画像、画像枚数、計算するチャンネル、マスク、ヒストグラム（出力）、
	//ヒストグラムの次元、ヒストグラムビンの下限
	calcHist(&srchist_img, 1, 0, Mat(), hist, 1, hdims, ranges);

	//最大値の計算
	minMaxLoc(hist, 0, &max_val);

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
		std::vector<std::vector<Point>> contors, contors2;
		threshold(gray_img, bin_img, levels1, 255, THRESH_TOZERO);
		findContours(bin_img, contors, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		imshow("bin image", bin_img);


		for (int i = 0; i < contors.size(); ++i){
			size_t count = contors[i].size();
			std::cout << "ellipse" << count << std::endl;
			Mat pointsf;
			Mat(contors[i]).convertTo(pointsf, CV_32F);

			if (count > 370 && count < 1000)
			{
				std::cout << "big ellipse" << std::endl;
				//楕円フィッティング
				RotatedRect box = fitEllipse(pointsf);
				//楕円の描画 大きい円はRED
				ellipse(src_img, box, Scalar(0, 0, 255), 2, CV_AA);
			}
			else if (count >20 && count < 100)
			{
				std::cout << "small ellipse!" << std::endl;
				//楕円フィッティング
				RotatedRect box = fitEllipse(pointsf);
				//楕円の描画 小さい円はBLUE
				ellipse(src_img, box, Scalar(255, 0, 0), 2, CV_AA);
			}
			else if (count < 20 || count > 1000)
			{
				continue;
			}
		}
		levels1++;
		if (levels1 == 170) {
			levels1 = 140;
		}
		namedWindow("Histogram", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		namedWindow("fit ellipse", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		namedWindow("bin image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		imshow("fit ellipse", src_img);
		imshow("bin image", bin_img);
		imshow("Histogram", hist_img);
		waitKey(100);
	}
}


