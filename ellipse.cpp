#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include<iostream>
#include<vector>

using namespace cv;
#define THRESH_MAX 255


char trackbarNameThreshold1[] = "Threshold1";//@comment トラックバーの名前

int main(){
	Mat gray_img, bin_img, dst_img, laplacian_img,canny_img;

	Mat src_img = imread("../snap3.png", 1); //@comment 3チャンネルカラー画像
	Mat srchist_img = imread("../snap3.png", 0);//@comment グレースケール画像
	if (!src_img.data)return-1;
	if (!srchist_img.data)return-1;


	resize(src_img, src_img, Size(), 0.3, 0.3);//@comment 元画像のリサイズ
	namedWindow("src",1);
	imshow("src",src_img);
	GaussianBlur(src_img, src_img, Size(5, 5), 2, 2);//@comment ガウシアンフィルタ
	//Laplacian(src_img, laplacian_img, CV_8UC1,3);//@comment ラプラシアンフィルタ
	//convertScaleAbs(laplacian_img, laplacian_img, 1, 0);

	cvtColor(src_img, gray_img, CV_BGR2GRAY);//@comment グレースケール変換

	//@comment ヒストグラムを描画する画像割り当て
	const int ch_width = 400, ch_height = 300;
	Mat hist_img(Size(ch_width, ch_height), CV_8UC3, Scalar::all(255));

	Mat hist;
	const int hdims[] = { 256 };
	const float hranges[] = { 0, 256 };
	const float* ranges[] = { hranges };
	double max_val = .0;
	namedWindow("bin image", 1);

	int levels1 = 140; //@comment 2値化閾値の設定
	createTrackbar(trackbarNameThreshold1, "bin image", &levels1, THRESH_MAX);

	//@comment シングルチャンネルのヒストグラム計算
	//@comment 画像、画像枚数、計算するチャンネル、マスク、ヒストグラム（出力）、
	//@comment ヒストグラムの次元、ヒストグラムビンの下限
	calcHist(&srchist_img, 1, 0, Mat(), hist, 1, hdims, ranges);

	//@comment 輝度の最大値の計算
	minMaxLoc(hist, 0, &max_val);

	while (1){
		calcHist(&srchist_img, 1, 0, Mat(), hist, 1, hdims, ranges);

		//@comment 輝度の最大値の計算
		minMaxLoc(hist, 0, &max_val);

		//@comment ヒストグラムのスケーリングと描画
		Scalar color = Scalar::all(100);
		//@comment スケーリング
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
		
		//imwrite("test.png", bin_img);
		findContours(bin_img, contors, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		//imwrite("test2.png", bin_img);
		cv::Canny(bin_img, canny_img, 50, 200, 3);
		imshow("bin image", bin_img);


		for (int i = 0; i < contors.size(); ++i){
			size_t count = contors[i].size();
			//std::cout << "ellipse" << count << std::endl;
			Mat pointsf;
			Mat(contors[i]).convertTo(pointsf, CV_32F);

			if (count > 370 && count < 1000)//@comment 大きい楕円の検出
			{
				std::cout << "big ellipse" <<" " <<count << std::endl;
				//@comment 楕円フィッティング
				RotatedRect box = fitEllipse(pointsf);
				//@comment 楕円の描画 大きい円はRED
				ellipse(src_img, box, Scalar(0, 0, 255), 2, CV_AA);
			}
			else if (count >30 && count < 100)//@comment 小さい楕円の検出
			{
				std::cout << "small ellipse!" << std::endl;
				//@comment 楕円フィッティング
				RotatedRect box = fitEllipse(pointsf);
				//@comment 楕円の描画 小さい円はBLUE
				ellipse(src_img, box, Scalar(255, 0, 0), 2, CV_AA);
			}
			else if (count < 20 || count > 1000) //@comment 大きすぎ、小さすぎる楕円は対象外
			{
				continue;
			}
		}
		levels1++;
		if (levels1 == 180) {
			levels1 = 145;
		}
		namedWindow("Histogram", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		namedWindow("fit ellipse", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		namedWindow("bin image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		//namedWindow("lap", 1);
		//imshow("lap", laplacian_img);
		imshow("fit ellipse", src_img);
		imshow("bin image", bin_img);
		imshow("Histogram", hist_img);
		namedWindow("canny", 1);
		imshow("canny",canny_img);
		if (waitKey(100) == 27)
		{
			//imwrite("test.png2",src_img);
			return 0;
		}
	}
}


