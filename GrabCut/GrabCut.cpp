#include "GrabCut.h"
#include "graph.h"
#include <math.h>
using namespace cv;
using namespace std;
GrabCut2D::~GrabCut2D(void)
{
}


enum {
	FORE_GROUND,
	BACK_GROUND,
};

static void initeGMM(cv::InputArray _img, cv::InputArray _mask, cv::OutputArray _model, int K, int back_or_fore);
static void encodeGMM(vector<Point3d> &mu, vector<Mat> &sigma, vector<double> &weight, OutputArray _model);
static void decodeGMM(vector<Point3d> &mu, vector<Mat> &sigma, vector<double> &weight, InputArray _model);

static void refineModel(InputArray _img, InputArray _mask, InputOutputArray _bgdModel, InputOutputArray _fgdModel);
static void assignGMMComponnent(InputArray _img, InputArray _mask, InputArray _bgdModel, InputArray _fgdModel, OutputArray _k_of_pixes); 
static void invertCovariance(InputArray _src, OutputArray _dst);
static int mostPossibleComponent(vector<Point3d>& mu, vector<Mat>& inverted_sigma, vector<double>& det, vector<double>& weight, Vec3b pix);
static double getGaussPossbility(Point3d mu, InputArray _inverted_sigma, double sigma_det, double weight, Vec3b pix);
static  void learnGMM(InputArray _img, InputArray _mask, OutputArray _bgdModel, OutputArray _fgdModel, InputArray _k_of_pixes, int K);
static void graphCut(InputArray _img, InputOutputArray _mask, InputArray _bgdModel, InputArray _fgdModel, Rect rect);
static double getGMMPossibility(vector<Point3d>& mu, vector<Mat>& inverted_sigma, vector<double>& det, vector<double>& weight, Vec3b pix);
static void print_err(char *s);
static double getSDD(InputArray _img);
static double getNLinkWeight(Vec3b a, Vec3b b, double dis, double gamma, double beta);

void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, 
	cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
    std::cout<<"Execute GrabCut Function: Please finish the code here!"<<std::endl;
	
//一.参数解释：
	//输入：
	 //cv::InputArray _img,     :输入的color图像(类型-cv:Mat)
     //cv::Rect rect            :在图像上画的矩形框（类型-cv:Rect) 
  	//int iterCount :           :每次分割的迭代次数（类型-int)


	//中间变量
	//cv::InputOutputArray _bgdModel ：   背景模型（推荐GMM)（类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）
	//cv::InputOutputArray _fgdModel :    前景模型（推荐GMM) （类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）


	//输出:
	//cv::InputOutputArray _mask  : 输出的分割结果 (类型： cv::Mat)

//二. 伪代码流程：
	//1.Load Input Image: 加载输入颜色图像;
	//2.Init Mask: 用矩形框初始化Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3）,矩形框以外设置为确定背景，矩形框以内设置为可能前景;
	//3.Init GMM: 定义并初始化GMM(其他模型完成分割也可得到基本分数，GMM完成会加分）
	//4.Sample Points:前背景颜色采样并进行聚类（建议用kmeans，其他聚类方法也可)
	//5.Learn GMM(根据聚类的样本更新每个GMM组件中的均值、协方差等参数）
	//6.Construct Graph（计算t-weight(数据项）和n-weight（平滑项））
	//7.Estimate Segmentation(调用maxFlow库进行分割)
	//8.Save Result输入结果（将结果mask输出，将mask中前景区域对应的彩色图像保存和显示在交互界面中）
	
	Mat img = _img.getMat();
	const int k = 5;
	if (_bgdModel.empty()){
		initeGMM(_img, _mask, _bgdModel, k, BACK_GROUND);
	}
	if (_fgdModel.empty()){
		initeGMM(_img, _mask, _fgdModel, k, FORE_GROUND);
	}

#ifdef _DEBUG
	Mat bgdModel = _bgdModel.getMat();
	Mat fgdModel = _fgdModel.getMat();
#endif // _DEBUG

	refineModel(_img, _mask, _bgdModel, _fgdModel);
	graphCut(_img, _mask, _bgdModel, _fgdModel, rect);
}


static void print_err(char *s){
	cerr << s << endl;
}

static void graphCut(InputArray _img, InputOutputArray _mask, InputArray _bgdModel, InputArray _fgdModel, Rect rect){
	Mat img = _img.getMat();
	Mat mask = _mask.getMat();

	vector<Point3d> fgd_mu, bgd_mu;
	vector<Mat> fgd_sigma, bgd_sigma;
	vector<double> fgd_weight, bgd_weight;
	decodeGMM(fgd_mu, fgd_sigma, fgd_weight, _fgdModel);
	decodeGMM(bgd_mu, bgd_sigma, bgd_weight, _bgdModel);

	//do some preparation
	int K = fgd_mu.size();
	vector<double> bgd_determinant(K), fgd_determinant(K);
	vector<Mat> bgd_inverted_sigma(K);
	vector<Mat> fgd_inverted_sigma(K);
	for (size_t i = 0; i < K; i++){
		bgd_determinant[i] = determinant(bgd_sigma[i]);
		fgd_determinant[i] = determinant(fgd_sigma[i]);
		invert(bgd_sigma[i], bgd_inverted_sigma[i]);
		invert(fgd_sigma[i], fgd_inverted_sigma[i]);
//#ifdef _DEBUG
//		Mat bgd_origin = bgd_sigma[i];
//		Mat bgd_inverted = bgd_inverted_sigma[i];
//		Mat fgd_origin = fgd_sigma[i];
//		Mat fgd_inverted = fgd_inverted_sigma[i];
//		Mat bgd_result = bgd_sigma[i] * bgd_inverted_sigma[i];
//		Mat fgd_result = fgd_sigma[i] * fgd_inverted_sigma[i];
//#endif // _DEBUG
	}

	//construct graph
	typedef Graph<double, double, double> GCGraph;
	GCGraph *g = new GCGraph(
		rect.area(), /*estimated # of nodes*/
		4 * rect.area(), /*estimated # of edges*/
		print_err);
	g->add_node(rect.area());

#ifdef _DEBUG
	Mat bgd_fgd = Mat::zeros(img.size(), CV_8U);
	Mat bgd_possibilities = Mat::zeros(img.size(), CV_64F);
	Mat fgd_possibilities = Mat::zeros(img.size(), CV_64F);
#endif

	//add t-link
	for (size_t i = 0; i < rect.height; i++){
		int img_row_num = rect.y + i;
		Vec3b* p_img = img.ptr<Vec3b>(img_row_num);
		uchar* p_mask = mask.ptr<uchar>(img_row_num);
#ifdef _DEBUG
		uchar* p_bgd_fgd = bgd_fgd.ptr<uchar>(img_row_num);
		double* p_fgd_pos = fgd_possibilities.ptr<double>(img_row_num);
		double* p_bgd_pos = bgd_possibilities.ptr<double>(img_row_num);
#endif
		for (size_t j = 0; j < rect.width; j++){
			int img_col_num = rect.x + j;
			double bgd_possibility = getGMMPossibility(bgd_mu, bgd_inverted_sigma, bgd_determinant, bgd_weight, p_img[img_col_num]);
			double fgd_possibility = getGMMPossibility(fgd_mu, fgd_inverted_sigma, fgd_determinant, fgd_weight, p_img[img_col_num]);
			if (p_mask[img_col_num] == 0){ //bgd
				bgd_possibility = 1;
				fgd_possibility = 0;
			}
			if (p_mask[img_col_num] == 1){ //fgd
				fgd_possibility = 1;
				bgd_possibility = 0;
			}
			g->add_tweights(i*rect.width + j, -log(bgd_possibility), -log(fgd_possibility)); //source -> fgd, sink -> bgd

#ifdef _DEBUG
			p_fgd_pos[img_col_num] = -log(fgd_possibility);
			p_bgd_pos[img_col_num] = -log(bgd_possibility);
			p_bgd_fgd[img_col_num] = fgd_possibility > bgd_possibility;
#endif
		}
	}

	//add n-link
	const double GAMMA = 3;
	double BETA = BETA = 1 / (4 * getSDD(img));
	for (size_t i = 0; i < rect.height; i++){
		int img_row_num = rect.y + i;
		Vec3b* p = img.ptr<Vec3b>(img_row_num);
		Vec3b* p_next_line;
		if ( i < rect.height -1 )
			p_next_line = img.ptr<Vec3b>(img_row_num + 1);
		for (size_t j = 0; j < rect.width; j++){
			int img_col_num = rect.x + j;
			double weight;
			int current_node = i*rect.width + j;
			if (j < rect.width - 1){
				weight = getNLinkWeight(p[img_col_num], p[img_col_num + 1], 1, GAMMA, BETA);
				g->add_edge(current_node, current_node + 1, weight, weight);
			}
			if ( i < rect.height - 1){
				weight = getNLinkWeight(p[img_col_num], p_next_line[img_col_num], 1, GAMMA, BETA);
				g->add_edge(current_node, current_node + rect.width, weight, weight);
				if (j < rect.width - 1){
					weight = getNLinkWeight(p[img_col_num], p_next_line[img_col_num + 1], sqrt(2), GAMMA, BETA);
					g->add_edge(current_node, current_node + rect.width + 1, weight, weight);
				}
				if (j > 0){
					weight = getNLinkWeight(p[img_col_num], p_next_line[img_col_num - 1], sqrt(2), GAMMA, BETA);
					g->add_edge(current_node, current_node + rect.width - 1, weight, weight);
				}
			}

		}
	}

	//max flow
	double flow = g->maxflow();

	//update mask
	for (size_t i = 0; i < rect.height; i++){
		int img_row_num = rect.y + i;
		uchar* p = mask.ptr<uchar>(img_row_num);
		for (size_t j = 0; j < rect.width; j++){
			int img_col_num = rect.x + j;
			if (g->what_segment(i*rect.width + j) == GCGraph::SOURCE
				&& p[img_col_num] == 2){
				p[img_col_num] = 3;
			}
			if (g->what_segment(i*rect.width + j) == GCGraph::SINK
				&& p[img_col_num] == 3){
				p[img_col_num] = 2;
			}
		}
	}
	delete g;
#ifdef _DEBUG
	Mat fgdModel = _fgdModel.getMat();
	Mat bgdModel = _bgdModel.getMat();
	Mat fgd_color(1, K, CV_8UC3);
	Mat bgd_color(1, K, CV_8UC3);
	Vec3b* p_fgd_color = fgd_color.ptr<Vec3b>(0);
	Vec3b* p_bgd_color = bgd_color.ptr<Vec3b>(0);
	double* p_fgdModel[3] = {
		fgdModel.ptr<double>(0),
		fgdModel.ptr<double>(1),
		fgdModel.ptr<double>(2),
	};
	double* p_bgdModel[3] = {
		bgdModel.ptr<double>(0),
		bgdModel.ptr<double>(1),
		bgdModel.ptr<double>(2),
	};
	for (size_t i = 0; i < K; i++){
		p_fgd_color[i] = Vec3b(
			p_fgdModel[0][i], 
			p_fgdModel[1][i], 
			p_fgdModel[2][i]);
		p_bgd_color[i] = Vec3b(
			p_bgdModel[0][i],
			p_bgdModel[1][i],
			p_bgdModel[2][i]);
	}
#endif // _DEBUG

}

static double getSDD(InputArray _img){
	Mat img = _img.getMat();
	Vec3b sum(0,0,0);
	for (size_t i = 0; i < img.rows; i++){
		Vec3b* p = img.ptr<Vec3b>(i);
		for (size_t j = 0; j < img.cols; j++){
			sum += p[j];
		}
	}
	Point3d mean(sum[0], sum[1], sum[2]);
	mean *= 1/(double)(img.cols*img.rows);
	double sdd = 0;
	for (size_t i = 0; i < img.rows; i++){
		Vec3b* p = img.ptr<Vec3b>(i);
		for (size_t j = 0; j < img.cols; j++){
			sdd += (p[j][0] - mean.x)*(p[j][0] - mean.x);
			sdd += (p[j][1] - mean.y)*(p[j][1] - mean.y);
			sdd += (p[j][2] - mean.z)*(p[j][2] - mean.z);
		}
	}
	sdd /= img.cols*img.rows;
	return sdd;
}

static double getNLinkWeight(Vec3b a, Vec3b b, double dis, double gamma, double beta){
	double norm = 0;
	Vec3b diff = (a - b);
	norm += diff[0] * diff[0];
	norm += diff[1] * diff[1];
	norm += diff[2] * diff[2];
	return gamma / dis * exp(-beta*norm);
}

static double getGMMPossibility(vector<Point3d>& mu, vector<Mat>& inverted_sigma, vector<double>& det, vector<double>& weight, Vec3b pix){
	double possibility = 0;
	double max_possibility = -1;
	for (size_t i = 0; i < mu.size(); i++){
		//possibility += getGaussPossbility(mu[i], inverted_sigma[i], det[i], weight[i], pix);
		possibility = getGaussPossbility(mu[i], inverted_sigma[i], det[i], weight[i], pix);
		if (max_possibility < possibility)
			max_possibility = possibility;
	}
	//return possibility;
	return max_possibility;
}

static void refineModel(InputArray _img, InputArray _mask, InputOutputArray _bgdModel, InputOutputArray _fgdModel){
	Mat k_of_pixes;
	assignGMMComponnent(_img, _mask, _bgdModel, _fgdModel, k_of_pixes);
	learnGMM(_img, _mask, _bgdModel, _fgdModel, k_of_pixes, _bgdModel.size().width);
}

static  void learnGMM(InputArray _img, InputArray _mask, OutputArray _bgdModel, OutputArray _fgdModel, InputArray _k_of_pixes, int K){
	Mat img = _img.getMat();
	Mat mask = _mask.getMat();
	Mat k_of_pixes = _k_of_pixes.getMat();

	/* calculate mean */
	vector<Point3d>
		fgd_mu(K, Point3d(0, 0, 0)),
		bgd_mu(K, Point3d(0, 0, 0));
	vector<int> fgd_count(K, 0), bgd_count(K, 0);
	int bgd_total_number = 0;
	int fgd_total_number = 0;
	for (size_t i = 0; i < img.rows; i++){
		Vec3b* p_img = img.ptr<Vec3b>(i);
		uchar* p_mask = mask.ptr<uchar>(i);
		uchar* p_k = k_of_pixes.ptr<uchar>(i);
		for (size_t j = 0; j < img.cols; j++){
			switch (p_mask[j]){
			case 0:
			case 2:
				bgd_mu[p_k[j]] += Point3d(p_img[j][0], p_img[j][1], p_img[j][2]);
				bgd_count[p_k[j]]++;
				break;
			case 1:
			case 3:
				fgd_mu[p_k[j]] += Point3d(p_img[j][0], p_img[j][1], p_img[j][2]);
				fgd_count[p_k[j]]++;
				break;
			default:
				cerr << __FILE__ << ':' << __LINE__ << " Unknown mask value " << p_mask[j] << endl;
				exit(1);
			}
		}
	}
	for (size_t i = 0; i < K; i++){
		bgd_mu[i] *= 1.0 / bgd_count[i];
		bgd_total_number += bgd_count[i];

		fgd_mu[i] *= 1.0 / fgd_count[i];
		fgd_total_number += fgd_count[i];
	}
	
	/* calculate covariance */
	vector<Mat>
		fgd_sigma(K),
		bgd_sigma(K);
	vector<double> fgd_weight(K), bgd_weight(K);
	vector<double*> fgd_sigma0(K), bgd_sigma0(K);
	vector<double*> fgd_sigma1(K), bgd_sigma1(K);
	vector<double*> fgd_sigma2(K), bgd_sigma2(K);
	for (size_t i = 0; i < K; i++){
		fgd_sigma[i] = Mat::zeros(3, 3, CV_64F);
		fgd_sigma0[i] = fgd_sigma[i].ptr<double>(0);
		fgd_sigma1[i] = fgd_sigma[i].ptr<double>(1);
		fgd_sigma2[i] = fgd_sigma[i].ptr<double>(2);

		bgd_sigma[i] = Mat::zeros(3, 3, CV_64F);
		bgd_sigma0[i] = bgd_sigma[i].ptr<double>(0);
		bgd_sigma1[i] = bgd_sigma[i].ptr<double>(1);
		bgd_sigma2[i] = bgd_sigma[i].ptr<double>(2);
	}
	for (size_t i = 0; i < img.rows; i++){
		Vec3b* p_img = img.ptr<Vec3b>(i);
		uchar* p_mask = mask.ptr<uchar>(i);
		uchar* p_k = k_of_pixes.ptr<uchar>(i);
		for (size_t j = 0; j < img.cols; j++){
			Point3d bgd_diff;
			Point3d fgd_diff;
			switch (p_mask[j]){
			case 0:
			case 2://bgd_
				bgd_diff = Point3d(p_img[j][0], p_img[j][1], p_img[j][2]) - bgd_mu[p_k[j]];
				bgd_sigma0[p_k[j]][0] += bgd_diff.x*bgd_diff.x; 
				bgd_sigma0[p_k[j]][1] += bgd_diff.x*bgd_diff.y;
				bgd_sigma0[p_k[j]][2] += bgd_diff.x*bgd_diff.z;

				bgd_sigma1[p_k[j]][0] += bgd_diff.y*bgd_diff.x; 
				bgd_sigma1[p_k[j]][1] += bgd_diff.y*bgd_diff.y; 
				bgd_sigma1[p_k[j]][2] += bgd_diff.y*bgd_diff.z;

				bgd_sigma2[p_k[j]][0] += bgd_diff.z*bgd_diff.x; 
				bgd_sigma2[p_k[j]][1] += bgd_diff.z*bgd_diff.y; 
				bgd_sigma2[p_k[j]][2] += bgd_diff.z*bgd_diff.z;
				break;
			case 1:
			case 3://fgd
				fgd_diff = Point3d(p_img[j][0], p_img[j][1], p_img[j][2]) - fgd_mu[p_k[j]];
				fgd_sigma0[p_k[j]][0] += fgd_diff.x*fgd_diff.x;
				fgd_sigma0[p_k[j]][1] += fgd_diff.x*fgd_diff.y;
				fgd_sigma0[p_k[j]][2] += fgd_diff.x*fgd_diff.z;

				fgd_sigma1[p_k[j]][0] += fgd_diff.y*fgd_diff.x;
				fgd_sigma1[p_k[j]][1] += fgd_diff.y*fgd_diff.y;
				fgd_sigma1[p_k[j]][2] += fgd_diff.y*fgd_diff.z;

				fgd_sigma2[p_k[j]][0] += fgd_diff.z*fgd_diff.x;
				fgd_sigma2[p_k[j]][1] += fgd_diff.z*fgd_diff.y;
				fgd_sigma2[p_k[j]][2] += fgd_diff.z*fgd_diff.z;
				break;
			default:
				cerr << __FILE__ << ':' << __LINE__ << " Unknown mask value " << p_mask[j] << endl;
				exit(1);
			}
		}
	}
	for (size_t i = 0; i < K; i++){
		bgd_sigma[i] /= bgd_count[i];
		bgd_weight[i] = bgd_count[i] * 1.0 / bgd_total_number;

		fgd_sigma[i] /= fgd_count[i];
		fgd_weight[i] = fgd_count[i] * 1.0 / fgd_total_number;
	}

	encodeGMM(fgd_mu, fgd_sigma, fgd_weight, _fgdModel);
	encodeGMM(bgd_mu, bgd_sigma, bgd_weight, _bgdModel);

#ifdef _DEBUG
	Mat bgdModel = _bgdModel.getMat();
	Mat fgdModel = _fgdModel.getMat();
#endif // _DEBUG
}

static void assignGMMComponnent(InputArray _img, InputArray _mask, InputArray _bgdModel, InputArray _fgdModel, OutputArray _k_of_pixes){
	Mat img = _img.getMat();
	Mat mask = _mask.getMat();
	_k_of_pixes.create(_img.size(), CV_8UC1);
	Mat k_of_pixes = _k_of_pixes.getMat();

	vector<Point3d> fgd_mu, bgd_mu;
	vector<Mat> fgd_sigma, bgd_sigma;
	vector<double> fgd_weight, bgd_weight;
	decodeGMM(fgd_mu, fgd_sigma, fgd_weight, _fgdModel);
	decodeGMM(bgd_mu, bgd_sigma, bgd_weight, _bgdModel);

#ifdef _DEBUG
	Mat fgdModel = _fgdModel.getMat();
	Mat bgdModel = _bgdModel.getMat();
	//Mat fgd_sigam0 = fgd_sigma[0];
	//Mat fgd_sigma1 = fgd_sigma[1];
	//Mat fgd_sigma2 = fgd_sigma[2];
	//Mat fgd_sigma3 = fgd_sigma[3];
	//Mat fgd_sigma4 = fgd_sigma[4];

	//Mat bgd_sigam0 = bgd_sigma[0];
	//Mat bgd_sigma1 = bgd_sigma[1];
	//Mat bgd_sigma2 = bgd_sigma[2];
	//Mat bgd_sigma3 = bgd_sigma[3];
	//Mat bgd_sigma4 = bgd_sigma[4];
#endif // _DEBUG

	//do some preparation
	int K = fgd_mu.size();
	vector<double> bgd_determinant(K), fgd_determinant(K);
	vector<Mat> bgd_inverted_sigma(K);
	vector<Mat> fgd_inverted_sigma(K);
	for (size_t i = 0; i < K; i++){
		bgd_determinant[i] = determinant(bgd_sigma[i]);
		fgd_determinant[i] = determinant(fgd_sigma[i]);
		invert(bgd_sigma[i], bgd_inverted_sigma[i]);
		invert(fgd_sigma[i], fgd_inverted_sigma[i]);
#ifdef _DEBUG
		Mat bgd_origin = bgd_sigma[i];
		Mat bgd_inverted = bgd_inverted_sigma[i];
		Mat fgd_origin = fgd_sigma[i];
		Mat fgd_inverted = fgd_inverted_sigma[i];
		Mat bgd_result = bgd_sigma[i] * bgd_inverted_sigma[i];
		Mat fgd_result = fgd_sigma[i] * fgd_inverted_sigma[i];
#endif // _DEBUG
	}

	for (size_t i = 0; i < mask.rows; i++){
		Vec3b* p_img = img.ptr<Vec3b>(i);
		uchar* p_mask = mask.ptr<uchar>(i);
		uchar* p_k = k_of_pixes.ptr<uchar>(i);
		for (size_t j = 0; j < mask.cols; j++){
			switch (p_mask[j]){
			case 0:
			case 2:
				p_k[j] = mostPossibleComponent(bgd_mu, bgd_inverted_sigma, bgd_determinant, bgd_weight, p_img[j]);
				break;
			case 1:
			case 3:
				p_k[j] = mostPossibleComponent(fgd_mu, fgd_inverted_sigma, fgd_determinant, fgd_weight, p_img[j]);
				break;
			default:
				cerr << __FILE__ << ':' << __LINE__ << " Unknown mask value " << p_mask[j] << endl;
				exit(1);
			}
		}
	}
}

static void invertCovariance(InputArray _src, OutputArray _dst){
	Mat src = _src.getMat();
	for (size_t i = 0; i < 3; i++){
		double &d = src.at<double>(i, i);
		if (d == 0)
			d = 1e-300;
	}
	invert(_src, _dst);
}

static int mostPossibleComponent(vector<Point3d>& mu, vector<Mat>& inverted_sigma, vector<double>& det, vector<double>& weight, Vec3b pix){
	double max_possibility = -1;
	int k;
#ifdef _DEBUG
	if (pix == Vec3b(255, 255, 255))
		k = 0;
#endif // _DEBUG

	for (size_t i = 0; i < mu.size(); i++){
		double possibility = getGaussPossbility(mu[i], inverted_sigma[i], det[i], weight[i], pix);
		if (possibility > max_possibility){
			max_possibility = possibility;
			k = i;
		}
	}
	return k;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static double getGaussPossbility(Point3d mu, InputArray _inverted_sigma, double sigma_det, double weight, Vec3b pix){
	//Mat sigma = _sigma.getMat();
	//double sigma_det = determinant(sigma);
	Mat inverted_sigma = _inverted_sigma.getMat();
	//invert(sigma, inverted_sigma);

	Vec3d diff(pix[0] - mu.x, pix[1] - mu.y, pix[2] - mu.z);

	double exponent = 0;
	for (size_t i = 0; i < 3; i++){
		double *p = inverted_sigma.ptr<double>(i);
		for (size_t j = 0; j < 3; j++){
			exponent += p[j] * diff[i] * diff[j];
		}
	}
	
	return weight / sqrt(8 * M_PI*M_PI*M_PI*sigma_det) * exp(-0.5*exponent);
}

static void initeGMM(cv::InputArray _img, cv::InputArray _mask, cv::OutputArray _model, int K, int back_or_fore){
	Mat img = _img.getMat();
	Mat mask = _mask.getMat();

	/* pick out pixes */
	//vector < Point3f > points;
	//for (size_t i = 0; i < mask.rows; i++){
	//	Vec3b* p_img = img.ptr<Vec3b>(i);
	//	uchar* p_mask = mask.ptr<uchar>(i);
	//	for (size_t j = 0; j < mask.cols; j++){
	//		switch (p_mask[j]){
	//		case 0:
	//		case 2:
	//			if (back_or_fore == BACK_GROUND)
	//				points.push_back(Point3f(p_img[j][0], p_img[j][1], p_img[j][2]));
	//			break;
	//		case 1:
	//		case 3:
	//			if (back_or_fore == FORE_GROUND)
	//				points.push_back(Point3f(p_img[j][0], p_img[j][1], p_img[j][2]));
	//			break;
	//		default:
	//			cerr << __FILE__ << ':' << __LINE__ << " Unknown mask value " << p_mask[j] << endl;
	//			exit(1);
	//		}
	//	}
	//}	
	int pix_number = 0;
	for (size_t i = 0; i < mask.rows; i++){
		Vec3b* p_img = img.ptr<Vec3b>(i);
		uchar* p_mask = mask.ptr<uchar>(i);
		for (size_t j = 0; j < mask.cols; j++){
			switch (p_mask[j]){
			case 0:
			case 2:
				if (back_or_fore == BACK_GROUND)
					pix_number++;
				break;
			case 1:
			case 3:
				if (back_or_fore == FORE_GROUND)
					pix_number++;
				break;
			default:
				cerr << __FILE__ << ':' << __LINE__ << " Unknown mask value " << p_mask[j] << endl;
				exit(1);
			}
		}
	}
	Mat points(pix_number, 3, CV_32F);
	pix_number = 0;
	for (size_t i = 0; i < mask.rows; i++){
		Vec3b* p_img = img.ptr<Vec3b>(i);
		uchar* p_mask = mask.ptr<uchar>(i);
		for (size_t j = 0; j < mask.cols; j++){
			switch (p_mask[j]){
			case 0:
			case 2:
				if (back_or_fore == BACK_GROUND){
					float* p = points.ptr<float>(pix_number);
					p[0] = p_img[j][0];
					p[1] = p_img[j][1];
					p[2] = p_img[j][2];
					pix_number++;
				}
				break;
			case 1:
			case 3:
				if (back_or_fore == FORE_GROUND){
					float* p = points.ptr<float>(pix_number);
					p[0] = p_img[j][0];
					p[1] = p_img[j][1];
					p[2] = p_img[j][2];
					pix_number++;
				}
				break;
			default:
				cerr << __FILE__ << ':' << __LINE__ << " Unknown mask value " << p_mask[j] << endl;
				exit(1);
			}
		}
	}


	/* cluster */
	TermCriteria criteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 20, 1e-7);
	Mat bestLabels;
	kmeans(points, K, bestLabels, criteria, 2, KMEANS_RANDOM_CENTERS);
	if (! bestLabels.isContinuous()){
		cerr << __FILE__ << ':' << __LINE__ << " bestLabels Mat is not continuous!" << endl;
	}
	int* pk = bestLabels.ptr<int>(0);

	/* calculate mean */
	vector<Point3d> mu(K, Point3d(0, 0, 0));
	vector<int> count(K, 0);	//k*1,int
	int total_number = 0;
	//for (size_t i = 0; i < points.size(); i++){
	for (size_t i = 0; i < points.rows; i++){
		int k_i;
		if (bestLabels.isContinuous())
			k_i = pk[i];
		else
			k_i = bestLabels.at<long>(i, 0);
		count[k_i]++;
		//mu[k_i] += Point3d(points[i].x, points[i].y, points[i].z);
		float* p = points.ptr<float>(i);
		mu[k_i] += Point3d(p[0], p[1], p[2]);
	}
	for (size_t i = 0; i < K; i++){
		mu[i] = mu[i] * (1.0 / count[i]);
		total_number += count[i];
	}

	/* calculate covariance */
	vector<Mat> sigma(K);
	vector<double> weight(K);
	vector<double*> sigma0(K);
	vector<double*> sigma1(K);
	vector<double*> sigma2(K);
	for (size_t i = 0; i < K; i++){
		sigma[i] = Mat::zeros(3, 3, CV_64F);
		sigma0[i] = sigma[i].ptr<double>(0);
		sigma1[i] = sigma[i].ptr<double>(1);
		sigma2[i] = sigma[i].ptr<double>(2);
	}
	//for (size_t i = 0; i < points.size(); i++){
	for (size_t i = 0; i < points.rows; i++){
		int k_i;
		if (bestLabels.isContinuous())
			k_i = pk[i];
		else
			k_i = bestLabels.at<long>(i, 0);

		//Point3d diff = Point3d(points[i]) - mu[k_i];

		float* p = points.ptr<float>(i);
		Point3d diff = Point3d(p[0], p[1], p[2]) - mu[k_i];

		sigma0[k_i][0] += diff.x*diff.x; sigma0[k_i][1] += diff.x*diff.y; sigma0[k_i][2] += diff.x*diff.z;
		sigma1[k_i][0] += diff.y*diff.x; sigma1[k_i][1] += diff.y*diff.y; sigma1[k_i][2] += diff.y*diff.z;
		sigma2[k_i][0] += diff.z*diff.x; sigma2[k_i][1] += diff.z*diff.y; sigma2[k_i][2] += diff.z*diff.z;
	}
	for (size_t i = 0; i < K; i++){
		sigma[i] /= count[i];
		weight[i] = count[i] * 1.0 / total_number;
	}

	encodeGMM(mu, sigma, weight, _model);
}

static void encodeGMM(vector<Point3d> &mu, vector<Mat> &sigma, vector<double> &weight, OutputArray _model){
	int K = mu.size();
	_model.create(13, K, CV_64F);
	Mat model = _model.getMat();

	double *mu0 = model.ptr<double>(0);
	double *mu1 = model.ptr<double>(1);
	double *mu2 = model.ptr<double>(2);
	double *sigma00 = model.ptr<double>(3);
	double *sigma01 = model.ptr<double>(4);
	double *sigma02 = model.ptr<double>(5);
	double *sigma10 = model.ptr<double>(6);
	double *sigma11 = model.ptr<double>(7);
	double *sigma12 = model.ptr<double>(8);
	double *sigma20 = model.ptr<double>(9);
	double *sigma21 = model.ptr<double>(10);
	double *sigma22 = model.ptr<double>(11);
	double *w = model.ptr<double>(12);
	for (size_t i = 0; i < K; i++)
	{
		mu0[i] = mu[i].x;
		mu1[i] = mu[i].y;
		mu2[i] = mu[i].z;
		double *p[3] = {
			sigma[i].ptr<double>(0),
			sigma[i].ptr<double>(1),
			sigma[i].ptr<double>(2),
		};
		sigma00[i] = p[0][0];// == 0 ? 1e-308 : p[0][0];
		sigma01[i] = p[0][1];
		sigma02[i] = p[0][2];
		sigma10[i] = p[1][0];
		sigma11[i] = p[1][1];// == 0 ? 1e-308 : p[1][1];
		sigma12[i] = p[1][2];
		sigma20[i] = p[2][0];
		sigma21[i] = p[2][1];
		sigma22[i] = p[2][2];// == 0 ? 1e-308 : p[2][2];
		w[i] = weight[i];
		if (p[0][0] == 0 || p[1][1] == 0 || p[2][2] == 0){
			cerr << __FILE__ << ':' << __LINE__ << " One of the RGB's variance is 0 in cluster " << i << endl;
			exit(1);
		}
	}
}

static void decodeGMM(vector<Point3d> &mu, vector<Mat> &sigma, vector<double> &weight, InputArray _model){
	Mat model = _model.getMat();
	int K = model.cols;

	mu.resize(K);
	sigma.resize(K);
	weight.resize(K);

	double *mu0 = model.ptr<double>(0);
	double *mu1 = model.ptr<double>(1);
	double *mu2 = model.ptr<double>(2);
	double *sigma00 = model.ptr<double>(3);
	double *sigma01 = model.ptr<double>(4);
	double *sigma02 = model.ptr<double>(5);
	double *sigma10 = model.ptr<double>(6);
	double *sigma11 = model.ptr<double>(7);
	double *sigma12 = model.ptr<double>(8);
	double *sigma20 = model.ptr<double>(9);
	double *sigma21 = model.ptr<double>(10);
	double *sigma22 = model.ptr<double>(11);
	double *w = model.ptr<double>(12);
	for (size_t i = 0; i < K; i++)
	{
		mu[i] = Point3d(mu0[i], mu1[i], mu2[i]);
		sigma[i] = (Mat_<double>(3, 3) <<
			sigma00[i], sigma01[i], sigma02[i],
			sigma10[i], sigma11[i], sigma12[i],
			sigma20[i], sigma21[i], sigma22[i]);
		weight[i] = w[i];
	}
}