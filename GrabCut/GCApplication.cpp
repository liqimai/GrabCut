#include "GCApplication.h"

//Set value for the class
void GCApplication::reset()
{
	if( !mask.empty() )
		mask.setTo(Scalar::all(GC_BGD));
	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear();  prFgdPxls.clear();

	isInitialized = false;
	rectState = NOT_SET;    
	lblsState = NOT_SET;
	prLblsState = NOT_SET;
	iterCount = 0;
}

//Set image and window name
void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName  )
{
	if( _image.empty() || _winName.empty() )
		return;
	image = &_image;
	winName = &_winName;
	mask.create( image->size(), CV_8UC1);
	reset();
}

//Show the result image
void GCApplication::showImage() const
{
	if( image->empty() || winName->empty() )
		return;

	Mat res;
	Mat binMask;
	if( !isInitialized )
		image->copyTo( res );
	else
	{
		getBinMask( mask, binMask );
		image->copyTo( res, binMask );  //show the GrabCuted image
	}

	vector<Point>::const_iterator it;
	//Using four different colors show the point which have been selected
	for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )  
		circle( res, *it, radius, BLUE, thickness );
	for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )  
		circle( res, *it, radius, GREEN, thickness );
	for( it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it )
		circle( res, *it, radius, LIGHTBLUE, thickness );
	for( it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it )
		circle( res, *it, radius, PINK, thickness );

	//Draw the rectangle
	if( rectState == IN_PROCESS || rectState == SET )
		rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), RED, 2);

	imshow( *winName, res );
}


//Using rect initialize the pixel 
void GCApplication::setRectInMask()
{
	assert( !mask.empty() );
	mask.setTo( GC_BGD );   //GC_BGD == 0
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols-rect.x);
	rect.height = min(rect.height, image->rows-rect.y);
	(mask(rect)).setTo( Scalar(GC_PR_FGD) );    //GC_PR_FGD == 3 
}


void GCApplication::setLblsInMask( int flags, Point p, bool isPr )
{
	vector<Point> *bpxls, *fpxls;
	uchar bvalue, fvalue;
	if( !isPr ) //Points which are sure being FGD or BGD
	{
		bpxls = &bgdPxls;
		fpxls = &fgdPxls;
		bvalue = GC_BGD;    //0
		fvalue = GC_FGD;    //1
	}
	else    //Probably FGD or Probably BGD
	{
		bpxls = &prBgdPxls;
		fpxls = &prFgdPxls;
		bvalue = GC_PR_BGD; //2
		fvalue = GC_PR_FGD; //3
	}
	if( flags & BGD_KEY )
	{
		bpxls->push_back(p);
		circle( mask, p, radius, bvalue, thickness );   //Set point value = 2
	}
	if( flags & FGD_KEY )
	{
		fpxls->push_back(p);
		circle( mask, p, radius, fvalue, thickness );   //Set point value = 3
	}
}


//Mouse Click Function: flags work with CV_EVENT_FLAG 
void GCApplication::mouseClick( int event, int x, int y, int flags, void* )
{
	switch( event )
	{
	case CV_EVENT_LBUTTONDOWN: // Set rect or GC_BGD(GC_FGD) labels
		{
			bool isb = (flags & BGD_KEY) != 0,
				isf = (flags & FGD_KEY) != 0;
			if( rectState == NOT_SET && !isb && !isf )//Only LEFT_KEY pressed
			{
				rectState = IN_PROCESS; //Be drawing the rectangle
				rect = Rect( x, y, 1, 1 );
			}
			if ( (isb || isf) && rectState == SET ) //Set the BGD/FGD(labels).after press the "ALT" key or "SHIFT" key,and have finish drawing the rectangle
			lblsState = IN_PROCESS;
		}
		break;
	case CV_EVENT_RBUTTONDOWN: // Set GC_PR_BGD(GC_PR_FGD) labels
		{
			bool isb = (flags & BGD_KEY) != 0,
				isf = (flags & FGD_KEY) != 0;
			if ( (isb || isf) && rectState == SET ) //Set the probably FGD/BGD labels
				prLblsState = IN_PROCESS;
		}
		break;
	case CV_EVENT_LBUTTONUP:
		if( rectState == IN_PROCESS )
		{
			rect = Rect( Point(rect.x, rect.y), Point(x,y) );   //After draw the rectangle
			rectState = SET;
			setRectInMask();
			assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
			showImage();
		}
		if( lblsState == IN_PROCESS )   
		{
			setLblsInMask(flags, Point(x,y), false);    // Draw the FGD points
			lblsState = SET;
			showImage();
		}
		break;
	case CV_EVENT_RBUTTONUP:
		if( prLblsState == IN_PROCESS )
		{
			setLblsInMask(flags, Point(x,y), true); //Draw the BGD points
			prLblsState = SET;
			showImage();
		}
		break;
	case CV_EVENT_MOUSEMOVE:
		if( rectState == IN_PROCESS )
		{
			rect = Rect( Point(rect.x, rect.y), Point(x,y) );
			assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
			showImage();   //Continue showing image
		}
		else if( lblsState == IN_PROCESS )
		{
			setLblsInMask(flags, Point(x,y), false);
			showImage();
		}
		else if( prLblsState == IN_PROCESS )
		{
			setLblsInMask(flags, Point(x,y), true);
			showImage();
		}
		break;
	}
}

//Execute GrabCut algorithm，and return the iter count.
int GCApplication::nextIter()
{
	Mat _image = *image;
	Mat _mask = mask;
	Rect _rect = rect;
	Mat _bgdModel = bgdModel;
	Mat _fgdModel = fgdModel;

	if( isInitialized )
		gc.GrabCut(*image, mask, rect, bgdModel, fgdModel,1,GC_CUT);
	else
	{
		if( rectState != SET )
			return iterCount;

		if( lblsState == SET || prLblsState == SET )
		 gc.GrabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_WITH_MASK );
		 else
		 gc.GrabCut(*image, mask, rect, bgdModel, fgdModel,1,GC_WITH_RECT);
		isInitialized = true;
	}
	iterCount++;

	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear(); prFgdPxls.clear();

	/*这里，已经完成分割了。你们应该在这里调用matting的函数
	 *分割的结果存在mask中
	 *Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3)
	 *你们可是使用mask和image，但是不要更改里面的值。
	 *另外，GCApplication::showImage()函数可能也要改一下。
	*/

 	return iterCount;
}

void GCApplication::borderMatting(){
	Mat _image = *image;
	
	Mat maskShow(mask.rows, mask.cols, CV_8U);
	for (int i = 0; i < mask.rows; i++){
		for (int j = 0; j < mask.cols; j++){
			if (mask.at<uchar>(i, j) == 3)
				maskShow.at<uchar>(i, j) = 255;
			else
				maskShow.at<uchar>(i, j) = 0;
		}
	}

	for (int i = 0; i < mask.rows; i++){
		for (int j = 0; j < mask.cols; j++){
			if (mask.at<uchar>(i, j) != 3){
				maskShow.at<uchar>(i, j) = 0;
			}
		}
	}

	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	/// Detect edges using canny
	Canny(maskShow, canny_output, -100, 100, 3);
	//imwrite("border.png", canny_output);
	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	cout << contours.size() << endl;
	/// Draw contours
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

	///parameters
	int w = 6;
	int L = 20;
	int deltaLevel = 30;
	int sigmaLevel = 10;
	Mat distance = Mat::ones(canny_output.size(), CV_32FC1) * 100;	
	Mat drawingTemp = Mat::zeros(canny_output.size(), CV_8UC1);
	Mat alpha = Mat::zeros(canny_output.size(), CV_32FC1);
	for (int i = 0; i < contours.size(); i++){
		if (contours[i].size() < 5) continue;
		cout << "contour" << i << " size: " << contours[i].size() << endl;
		Mat index = Mat::ones(canny_output.size(), CV_32SC1) * (-1);
		vector<vector<Point2i> > pointVector;
		for (int j = 0; j < contours[i].size(); j++){
			Point2i p;
			p.x = contours[i][j].y;
			p.y = contours[i][j].x;
			vector<Point2i> point;
			point.push_back(p);
			pointVector.push_back(point);
			index.at<int>(p.x, p.y) = j;
			drawingTemp.at<uchar>(p.x, p.y) = 255;
		}
		//cout << "PointVector:" << pointVector.size();
		Scalar color = Scalar(255, 255, 255);
		drawContours(drawing, contours, i, color, 2 * w + 1, 8, hierarchy, 0, Point());
		for (int ii = 0; ii < drawing.rows; ii++){
			for (int jj = 0; jj < drawing.cols; jj++){
				if (drawing.at<Vec3b>(ii,jj) == Vec3b(255, 255, 255)){
					double dist = 20 * w * w;
					int index1 = -1;
					for (int p = -w; p <= w; p++){
						for (int q = -w; q <= w; q++){
							if (ii + p < 0 || ii + p >= drawing.rows || jj + q < 0 || jj + q >= drawing.cols)
								continue;
							if (drawingTemp.at<uchar>(ii + p, jj + q) == 255){
								if (p*p + q*q < dist){
									index1 = index.at<int>(ii + p, jj + q);
									dist = p*p + q*q;
								}
							}
							
						}
					}
					
					index.at<int>(ii, jj) = index1;
					Point2i point = Point2i(ii, jj);
					if (index1 != -1)
						pointVector[index1].push_back(point);
				}
			}
		}
		for (int ii = 0; ii < drawing.rows; ii++){
			for (int jj = 0; jj < drawing.cols; jj++){
				distance.at<float>(ii, jj) = -pointPolygonTest(contours[i], Point2f(jj, ii), true);
			}
		}

		///计算均值和方差
		Mat uf = Mat::zeros(canny_output.size(), CV_32FC3); 
		Mat ub = Mat::zeros(canny_output.size(), CV_32FC3);
		Mat sigmaf = Mat::zeros(canny_output.size(), CV_32FC3);
		Mat sigmab = Mat::zeros(canny_output.size(), CV_32FC3);
		
		for (int ii = 0; ii < drawing.rows; ii++){
			for (int jj = 0; jj < drawing.cols; jj++){
				if (index.at<int>(ii, jj) != -1){
					int countf = 0, countb = 0;
					Vec3f ufTemp(0, 0, 0);
					Vec3f ubTemp(0, 0, 0);
					Vec3f sigmafTemp(0, 0, 0);
					Vec3f sigmabTemp(0, 0, 0);
					for (int p = -L; p <= L; p++){
						for (int q = -L; q <= L; q++){
							if (ii + p < 0 || ii + p >= drawing.rows || jj + q < 0 || jj + q >= drawing.cols)
								continue;
							if (maskShow.at<uchar>(ii + p, jj + q) == 0){
								countb++;
								ubTemp += _image.at<Vec3b>(ii + p, jj + q);
							}
							else{
								countf++;
								ufTemp += _image.at<Vec3b>(ii + p, jj + q);
							}
						}
					}
					ufTemp = ufTemp * 1.0 / countf;
					ubTemp = ubTemp * 1.0 / countb;
					uf.at<Vec3f>(ii, jj) = ufTemp;
					ub.at<Vec3f>(ii, jj) = ubTemp;
					for (int p = -L; p <= L; p++){
						for (int q = -L; q <= L; q++){
							if (ii + p < 0 || ii + p >= drawing.rows || jj + q < 0 || jj + q >= drawing.cols)
								continue;
							if (maskShow.at<uchar>(ii + p, jj + q) == 0){
								Vec3f temp;
								multiply((Vec3f(_image.at<Vec3b>(ii + p, jj + q)) - ubTemp), (Vec3f(_image.at<Vec3b>(ii + p, jj + q)) - ubTemp), temp);
								sigmabTemp += temp;
							}
							else{
								Vec3f temp;
								multiply((Vec3f(_image.at<Vec3b>(ii + p, jj + q)) - ufTemp), (Vec3f(_image.at<Vec3b>(ii + p, jj + q)) - ufTemp), temp);
								sigmafTemp += temp;
							}
						}
					}
					sigmabTemp /= countb;
					sigmafTemp /= countf;
					sigmab.at<Vec3f>(ii, jj) = sigmabTemp;
					sigmaf.at<Vec3f>(ii, jj) = sigmafTemp;
				}
			}
		}
		Mat ufShow, ubShow;
		uf.convertTo(ufShow, CV_8UC3);
		ub.convertTo(ubShow, CV_8UC3);
		//imwrite("uf.png", ufShow);
		//imwrite("ub.png", ubShow);

		///动态规划
		Mat alpha = Mat::zeros(canny_output.size(), CV_32FC1);
		Mat sigmaShow = Mat::zeros(canny_output.size(), CV_32FC1);
		Mat deltaShow = Mat::zeros(canny_output.size(), CV_32FC1);
		int method = 1;
		double lambda1 = 50, lambda2 = 1000;
		if (method == 1){
			vector<float> sigmaVector, deltaVector, dataVector;
			for (int t = 0; t < contours[i].size(); t++){
				float datamin = 100000000;
				float mindelta = 0, minsigma = 0;
				for (int p = 1; p <= deltaLevel; p++){
					for (int q = 1; q <= sigmaLevel; q++){
						float delta = 4 * w * 1.0 / p - 2 * w;
						float sigma = 2 * w * 1.0 / q;
						float D = 0;
						float data = 0;
						if (t != 0){
							data += lambda1*(delta - deltaVector[t - 1]) + lambda2*(sigma - sigmaVector[t - 1]);
						}
						for (int k = 0; k < pointVector[t].size(); k++){
							int x = pointVector[t][k].x;
							int y = pointVector[t][k].y;
							float alpha1 = 1.0 / (1 + exp((delta - distance.at<float>(x, y)) / sigma));
							Vec3f u = (1 - alpha1)*uf.at<Vec3f>(x, y) + alpha1*ub.at<Vec3f>(x, y);
							Vec3f sigmaSum = (1 - alpha1)*(1 - alpha1)*sigmaf.at<Vec3f>(x, y) + alpha1*alpha1*sigmab.at<Vec3f>(x, y);
							Vec3f temp = Vec3f(_image.at<Vec3b>(x, y)) - u;
							for (int d = 0; d < 3; d++){
								data += temp[d] * temp[d] / sigmaSum[d];
							}
						}
						if (data < datamin){
							datamin = data;
							mindelta = delta;
							minsigma = sigma;
						}
						
					}
				}
				sigmaVector.push_back(minsigma);
				deltaVector.push_back(mindelta);
				dataVector.push_back(datamin);
			}
			for (int t = 0; t < contours[i].size(); t++){
				float delta = deltaVector[t];
				float sigma = sigmaVector[t];
				for (int k = 0; k < pointVector[t].size(); k++){
					int x = pointVector[t][k].x;
					int y = pointVector[t][k].y;	
					float alpha1 = 1.0 / (1 + exp((delta - distance.at<float>(x, y)) / sigma));
					alpha.at<float>(x, y) = alpha1;
					sigmaShow.at<float>(x, y) = sigma;
					deltaShow.at<float>(x, y) = delta;
				}
			}

		}
	}

	///foreground estimation写在这里 
	/*
		alpha是alpha矩阵
		maskShow中0为论文中3部分分割出来的背景，255为前景。
	*/
}