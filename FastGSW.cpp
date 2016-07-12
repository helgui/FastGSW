#include "stdafx.h"
#include "FastGSW.h"
#include "Visualization.h"

using namespace cv;
using namespace std;

static int dr[4] = {-1, 0, -1, -1};
static int dc[4] = {-1, -1, 0, 1};

static int dx[8] = {-1, 1, 0, 0, 1, 1, -1, -1};
static int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};

#define pix(i, j) at<Vec3b>((i), (j))

class Node {
public:
	float d;
	int x, y;
	Node(float d, int x, int y)
		: d(d), x(x), y(y) {
	}
	Node(float d, const Point &pnt)
		:d(d), x(pnt.x), y(pnt.y) {
	}
	bool operator <(const Node &rhs) const {
		return d > rhs.d;
	}
};

inline float diff(const Vec3f &a, const Vec3f &b) {
	return (float)norm(a - b);
}

inline float psAt(const vector<vector<float>> &ps, int r, int c) {
	if (r < 0 || c < 0)
		return 0.0f;
	if (r >= (int)ps.size())
		r = ps.size() - 1;
	if (c >= (int)ps[0].size())
		c = ps[0].size() - 1;
	return ps[r][c];
}

inline float rectSum(const vector<vector<float>> &ps, int r1, int c1, int r2, int c2) {
	return psAt(ps, r2, c2) -
		psAt(ps, r2, c1 - 1) -
		psAt(ps, r1 - 1, c2) +
		psAt(ps, r1 - 1, c1 - 1);
}

inline bool in(int r, int c, const Mat &mat) {
	return r >= 0 && r < mat.rows && c >= 0 && c < mat.cols;
}

void FastGSW::computeLR(const Mat &left, const Mat &right, const Mat &seg, Mat &disp, int dispType) {
	disp.create(left.size(), dispType);

	int mxIdx = *max_element(seg.begin<int>(), seg.end<int>());
	
	vector<vector<Point>> segPts(mxIdx + 1);
	for (int i = 0; i < seg.rows; ++i) {
		for (int j = 0; j < seg.cols; ++j) {
			segPts[seg.at<int>(i, j)].emplace_back(j, i);
		}
	}
	vector<vector<float>> w(seg.rows, vector<float>(seg.cols, numeric_limits<float>::infinity()));
	
	vector<Vec3f> segColor(mxIdx + 1, Vec3f(0, 0, 0));
	
	for (int i = 0; i <= mxIdx; ++i) {
		geoMaskDijkstra(left, seg, segPts[i][segPts[i].size()/2], w);
		float denom = 0.0f;																							
		for (const Point &pnt : segPts[i]) {
			w[pnt.y][pnt.x] = exp(-w[pnt.y][pnt.x] / gamma);
			Vec3f tmp = left.pix(pnt.y, pnt.x);
			segColor[i] += w[pnt.y][pnt.x]*tmp;
			denom += w[pnt.y][pnt.x];
		}
		segColor[i] /= denom;
	}

	vector<vector<float>> cst(seg.rows, vector<float>(seg.cols, numeric_limits<float>::infinity()));
	vector<vector<float>> ps(seg.rows, vector<float>(seg.cols, 0.0f));

	for (int d = 0; d <= numDisp; ++d) {
		
		for (int i = 0; i < seg.rows; ++i) {
			for (int j = 0; j < seg.cols; ++j) {
				ps[i][j] = psAt(ps, i - 1, j) + psAt(ps, i, j - 1) - psAt(ps, i - 1, j - 1) + mcost(segColor[seg.at<int>(i, j)], right, i, j, d);
			}
		}

		for (int i = 0; i < seg.rows; ++i) {
			for (int j = 0; j < seg.cols; ++j) {
				float tc = cost(ps, i, j);
				if (tc < cst[i][j]) {
					cst[i][j] = tc;
					if (dispType == CV_16S)
						disp.at<short>(i, j) = (short)d;
					else
						disp.at<float>(i, j) = (float)d;
				}
			}
		}
	}
}

void FastGSW::computeRL(const Mat &left, const Mat &right, const Mat &seg, Mat &disp, int dispType) {
	disp.create(left.size(), dispType);
	//only for visualization vvv
	//Mat tmp = segRandomPaint(seg);
	//left.copyTo(tmp);
	//segContours(seg, tmp);
	//imwrite("segmentation.png", tmp);
	//only for visualization ^^^

	int mxIdx = *max_element(seg.begin<int>(), seg.end<int>());
	vector<vector<Point>> segPts(mxIdx + 1);
	for (int i = 0; i < seg.rows; ++i) {
		for (int j = 0; j < seg.cols; ++j) {
			segPts[seg.at<int>(i, j)].emplace_back(j, i);
		}
	}

	//Geodesic supports weights
	vector<vector<float>> w(seg.rows, vector<float>(seg.cols, 0.0f));

	//Segment color
	vector<Vec3f> segColor(mxIdx + 1, Vec3f(0, 0, 0));

	Mat display(seg.rows, seg.cols, CV_8UC1);

	for (int i = 0; i <= mxIdx; ++i) {
		geoMaskDijkstra(right, seg, segPts[i][segPts[i].size()/2], w);
		//only for visualization vvv
		//float maxw = 0.0f;
		//for (const Point &pnt : segPts[i]) {
		//	maxw = max(maxw, w[pnt.y][pnt.x]);
		//}
		//only for visualization ^^^
		float denom = 0.0f;
		for (const Point &pnt : segPts[i]) {
			w[pnt.y][pnt.x] = exp(-w[pnt.y][pnt.x] / gamma);
			//display.at<uchar>(pnt) = ((w[pnt.y][pnt.x] / maxw)*255.0f);
			Vec3f tmp = right.pix(pnt.y, pnt.x);
			segColor[i] += w[pnt.y][pnt.x] * tmp;
			denom += w[pnt.y][pnt.x];
		}
		segColor[i] /= denom;
		//only for visualization vvv
		//for (const Point &pnt : segPts[i]) {
		//	tmp.at<Vec3b>(pnt) = segColor[i];
		//}
		//only for visualization ^^^
	}

	vector<vector<float>> cst(seg.rows, vector<float>(seg.cols, numeric_limits<float>::infinity()));
	vector<vector<float>> ps(seg.rows, vector<float>(seg.cols, 0.0f));


	for (int d = 0; d <= numDisp; ++d) {
		//prefix sums
		for (int i = 0; i < seg.rows; ++i) {
			for (int j = 0; j < seg.cols; ++j) {
				ps[i][j] = psAt(ps, i - 1, j) + psAt(ps, i, j - 1) - psAt(ps, i - 1, j - 1) + mcost(segColor[seg.at<int>(i, j)], left, i, j, -d);
			}
		}

		for (int i = 0; i < seg.rows; ++i) {
			for (int j = 0; j < seg.cols; ++j) {
				float tc = cost(ps, i, j);
				if (tc < cst[i][j]) {
					cst[i][j] = tc;
					if (dispType == CV_16S)
						disp.at<short>(i, j) = (short)(-d);
					else
						disp.at<float>(i, j) = (float)(-d);
				}
			}
		}
	}
	//imwrite("recolored.png", tmp);
	//imwrite("geodesic.png", display);
}

float FastGSW::cost(const vector<vector<float>> &ps, int r, int c) {
	return rectSum(ps, r - W/2, c - W/2, r + W/2 , c + W/2);
}

void FastGSW::forwardDistSeg(const Mat &img, const Mat &seg, const vector<Point> &segPts, vector<float> &dist) {
	/*
	for (int i = 0; i < (int)segPts.size(); ++i) {
		const Point &pnt = segPts[i];
		for (int k = 0; k < 4; ++k) {
			if (!in(pnt.y + dr[k], pnt.x + dc[k], img))
				continue;
			if (seg.at<int>(pnt.y + dr[k], pnt.x + dc[k]) != seg.at<int>(segPts[0]))
				continue;
			int pos = pointPos(Point(pnt.x + dc[k], pnt.y + dr[k]), segPts);
			dist[i] = min(dist[i], dist[pos] + diff(img.pix(pnt.y + dr[k], pnt.x + dc[k]), img.pix(pnt.y, pnt.x)));
		}
	}
	*/
}

void FastGSW::backwardDistSeg(const Mat &img, const Mat &seg,  const vector<Point> &segPts, vector<float> &dist) {
	/*
	for (int i = segPts.size() - 1; i >= 0; --i) {
		const Point & pnt = segPts[i];
		for (int k = 0; k < 4; ++k) {
			if (!in(pnt.y - dr[k], pnt.x - dc[k], img))
				continue;
			if (seg.at<int>(pnt.y - dr[k], pnt.x - dc[k]) != seg.at<int>(segPts[0]))
				continue;
			int pos = pointPos(Point(pnt.x - dc[k], pnt.y - dr[k]), segPts);
			dist[i] = min(dist[i], dist[pos] + diff(img.pix(pnt.y - dr[k], pnt.x - dc[k]), img.pix(pnt.y, pnt.x)));
		}
	}
	*/
}

void FastGSW::geoMaskBorgefors(const Mat &left, const Mat &seg, const vector<Point> &segPts, vector<vector<float>> &w) {
	/*
	vector<float> dist(segPts.size(), numeric_limits<float>::infinity());
	dist[segPts.size() / 2] = 0.0f;
	for (int iter = 0; iter < 2; ++iter) {
		forwardDistSeg(left, seg, segPts, dist);
		backwardDistSeg(left, seg, segPts, dist);
	}
	for (int i = 0; i < (int)segPts.size(); ++i) {
		w[segPts[i].y][segPts[i].x] = exp(-(dist[i] / gamma));
	}
	*/
}

void FastGSW::geoMaskDijkstra(const Mat &left, const Mat &seg, const Point &pivot, vector<vector<float>> &w) {
	w[pivot.y][pivot.x] = 0.0f;
	priority_queue<Node> heap;
	heap.emplace(0.0f, pivot);
	int segIdx = seg.at<int>(pivot);
	int ny, nx;
	while (!heap.empty()) {
		Node v = heap.top();
		heap.pop();
		if (v.d > w[v.y][v.x])
			continue;
		for (int dy = -1; dy <= 1; ++dy) {
			for (int dx = -1; dx <= 1; ++dx) {
				if (dx == 0 && dy == 0)
					continue;
				ny = v.y + dy;
				nx = v.x + dx;
				if (!in(ny, nx, seg))
					continue;
				if (seg.at<int>(ny, nx) != segIdx)
					continue;
				if (w[ny][nx] > (v.d + diff(left.pix(v.y, v.x), left.pix(ny, nx)))) {
					w[ny][nx] = (v.d + diff(left.pix(v.y, v.x), left.pix(ny, nx)));
					heap.emplace(w[ny][nx], nx, ny);
				}
			}
		}
	}
}

float FastGSW::mcost(const Vec3f &segColor, const Mat &right, int r, int c, int d) {
	/*Outside points are penalized by predefined constant*/
	if (!in(r, c - d, right))
		return 60.0f;
	
	Vec3f pixColor = right.at<Vec3b>(r, c - d);
	return (float)norm(segColor - pixColor, NORM_L2);
}

float FastGSW::getGamma() const {
	return gamma;
}

int FastGSW::getWindowSize() const {
	return W;
}

int FastGSW::getNumDisp() const {
	return numDisp;
}

void FastGSW::setGamma(float newGamma) {
	gamma = newGamma;
}

void FastGSW::setWindowSize(int newWinSize) {
	W = newWinSize;
}

void FastGSW::setNumDisp(int newNumDisp) {
	numDisp = newNumDisp;
}

