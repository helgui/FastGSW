#pragma once
#ifndef FAST_STEREO_H
#define FAST_STEREO_H
#include <opencv2/core.hpp>

class FastGSW {
public:
	/*
	 * win - support window size (odd integer)
	 * gamma - scale factor to convert geodesic distance to support weight
	 * numDisp - maximum number of disparities
	 * minimum disparity is always 0
	*/
	
	FastGSW(int win, float gamma, int numDisp) :
		W(win), gamma(gamma), numDisp(numDisp) {
	}
	FastGSW() :
		FastGSW(31, 10.0, 70){
	}
	void computeLR(const cv::Mat &left, const cv::Mat &right, const cv::Mat &segmentation, cv::Mat &disp, int dispType = CV_32F);
	void computeRL(const cv::Mat &left, const cv::Mat &right, const cv::Mat &segmentation, cv::Mat &disp, int dispType = CV_32F);

	float getGamma() const;
	int getWindowSize() const;
	int getNumDisp() const;

	void setGamma(float newGamma);
	void setWindowSize(int newWinSize);
	void setNumDisp(int newNumDisp);
	
private:
	/*Forward iteration of Borgefors' algorithm*/
	void forwardDistSeg(const cv::Mat &img, const cv::Mat &seg, const std::vector<cv::Point> &segPts, std::vector<float> &dist);
	
	/*Backward iteration of Borgefors algorithm*/
	void backwardDistSeg(const cv::Mat &img, const cv::Mat &seg, const std::vector<cv::Point> &segPts, std::vector<float> &dist);
	
	/*Computes geodesic mask via Borgefors' algorithm*/
	void geoMaskBorgefors(const cv::Mat &left, const cv::Mat &seg, const std::vector<cv::Point> &segPts, std::vector<std::vector<float>> &w);
	
	/*Computes geodesic mask via Dijkstra's algorithm (with heap)*/
	void geoMaskDijkstra(const cv::Mat &left, const cv::Mat &seg, const cv::Point &pivot, std::vector<std::vector<float>> &w);

	float cost(const std::vector<std::vector<float>> &ps, int r, int c);

	
	/*Matching cost (Euclidean distance)*/
	float mcost(const cv::Vec3f &segColor, const cv::Mat &right, int r, int c, int d);
	
	int numDisp;
	int W;
	float gamma;
};
#endif
