#include "SGM.h"
#include <unistd.h>
#include <algorithm>
#include <vector>
#include "io_flow.h"
#include "opencv2/photo.hpp"
SGM::SGM(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_):
imgLeftLast(imgLeftLast_), imgLeft(imgLeft_), imgRight(imgRight_), PENALTY1(PENALTY1_), PENALTY2(PENALTY2_), winRadius(winRadius_)
{
	this->WIDTH  = imgLeft.cols;
	this->HEIGHT = imgLeft.rows;
	censusImageLeft	 = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	censusImageRight = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	censusImageLeftLast = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	cost = Mat::zeros(HEIGHT, WIDTH, CV_32FC(DISP_RANGE));
	costRight = Mat::zeros(HEIGHT, WIDTH, CV_32FC(DISP_RANGE));
	directCost = Mat::zeros(HEIGHT, WIDTH, CV_32SC(DISP_RANGE));
	accumulatedCost = Mat::zeros(HEIGHT, WIDTH, CV_32SC(DISP_RANGE));
};

void SGM::writeDerivative(){}

void SGM::computeCensus(const cv::Mat &image, cv::Mat &censusImg){


	for (int y = winRadius; y < HEIGHT - winRadius; ++y) {
		for (int x = winRadius; x < WIDTH - winRadius; ++x) {
			unsigned char centerValue = image.at<uchar>(y,x);

			int censusCode = 0;
			for (int neiY = -winRadius; neiY <= winRadius; ++neiY) {
				for (int neiX = -winRadius; neiX <= winRadius; ++neiX) {
					censusCode = censusCode << 1;
					if (image.at<uchar>(y + neiY, x + neiX) >= centerValue) censusCode += 1;		
				}
			}
			
			censusImg.at<uchar>(y,x) = static_cast<unsigned char>(censusCode);
		}
	}
}


int SGM::computeHammingDist(const uchar left, const uchar right){

	int var = static_cast<int>(left ^ right);
	int count = 0;

	while(var){
		var = var & (var - 1);
		count++;
	}
	return count;
}




void SGM::sumOverAllCost(){

	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			accumulatedCost.at<SGM::VecDf>(y,x) += directCost.at<SGM::VecDf>(y,x);
			
		}
	}
}



void SGM::createDisparity(cv::Mat &disparity){
	
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			float imax = std::numeric_limits<float>::max();
			int min_index = 0;
			SGM::VecDf vec = accumulatedCost.at<SGM::VecDf>(y,x);

			for(int d = 0; d < DISP_RANGE; d++){
				if(vec[d] < imax ){ imax = vec[d]; min_index = d;}
			}
			disparity.at<uchar>(y,x) = static_cast<uchar>(DIS_FACTOR*min_index);
			
			
		}
	}

}

void SGM::setPenalty(const int penalty_1, const int penalty_2){
	PENALTY1 = penalty_1;
	PENALTY2 = penalty_2;
}

void SGM::postProcess(cv::Mat &disparity){}

void SGM::consistencyCheck(cv::Mat disparityLeft, cv::Mat disparityRight, cv::Mat disparity, bool interpl){}	

void SGM::resetDirAccumulatedCost(){}

void SGM::resetDirAccumulatedCostLast(){}

void SGMFlow::resetDirAccumulatedCost(){

	for (int x = 0 ; x < WIDTH; x++ ) {
      		for ( int y = 0; y < HEIGHT; y++ ) {
			if(disFlag.at<uchar>(y,x)==static_cast<uchar>(DISFLAG)){
				for(int d = 0; d < DISP_RANGE; d++){
            				directCost.at<SGM::VecDf>(y,x)[d] = 0.0;
				}
			}
					      
        	}
      	}

}

void SGMFlow::resetDirAccumulatedCostLast(){

	for (int x = 0 ; x < WIDTH; x++ ) {
      		for ( int y = 0; y < HEIGHT; y++ ) {
			if(disFlagBackward.at<uchar>(y,x)==static_cast<uchar>(DISFLAG)){
				for(int d = 0; d < DISP_RANGE; d++){
            				directCost.at<SGM::VecDf>(y,x)[d] = 0.0;
				}
			}
					      
        	}
      	}
}

void SGM::runSGM(cv::Mat &disparity){

	std::cout<<"compute Census: ";
	computeCensus(imgLeft , censusImageLeft);
	computeCensus(imgLeftLast , censusImageLeftLast);
	computeCensus(imgRight, censusImageRight);
	std::cout<<"done"<<std::endl;
	std::cout<<"compute derivative: ";
	computeDerivative();
	std::cout<<"done"<<std::endl;
	std::cout<<"compute pixel-wise cost: ";
	computeCost();
	std::cout<<"done"<<std::endl;

	std::cout<<"aggregation starts:"<<std::endl;
	aggregation<1,0>(cost);
	resetDirAccumulatedCost();
	sumOverAllCost();

//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<0,1>(cost);
	resetDirAccumulatedCost();
	sumOverAllCost();

//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<0,-1>(cost);
	resetDirAccumulatedCost();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<-1,0>(cost);
	resetDirAccumulatedCost();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

/*	aggregation<1,1>(cost);
	resetDirAccumulatedCost();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;	
	aggregation<-1,1>(cost);
	resetDirAccumulatedCost();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;
	aggregation<1,-1>(cost);
	resetDirAccumulatedCost();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;
	aggregation<-1,-1>(cost);
	resetDirAccumulatedCost();
	sumOverAllCost();
*/
/*
	aggregation<2,1>();
	sumOverAllCost();

	aggregation<2,-1>();
	sumOverAllCost();

	aggregation<-2,-1>();
	sumOverAllCost();

	aggregation<-2,1>();
	sumOverAllCost();

	aggregation<1,2>();
	sumOverAllCost();

	aggregation<-1,2>();
	sumOverAllCost();

	aggregation<1,-2>();
	sumOverAllCost();

	aggregation<-1,-2>();
	sumOverAllCost();

	createDisparity(disparity);
	postProcess(disparity);

*/

	cv::Mat disparityLeft(HEIGHT, WIDTH, CV_8UC1);
	cv::Mat disparityTemp(HEIGHT, WIDTH, CV_8UC1);
	createDisparity(disparityTemp);
	fastNlMeansDenoising(disparityTemp, disparity);
	imwrite("./disparityForward.jpg", disparity);

/*
for(int y = 0; y < HEIGHT ; y++){
	for(int x = 0; x < WIDTH; x++){
		for(int d = 0; d < DISP_RANGE; d++){
			accumulatedCost.at<SGM::VecDf>(y,x)[d]=0.f;
			directCost.at<SGM::VecDf>(y,x)[d]=0.f;
		}
	}
}
*/

	std::cout<<"compute costRight: ";
	computeCostRight();
	std::cout<<"done"<<std::endl;

	std::cout<<"aggregation starts:"<<std::endl;
	aggregation<1,0>(costRight);
	resetDirAccumulatedCost();
	sumOverAllCost();

//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<0,1>(costRight);
	resetDirAccumulatedCostLast();
	sumOverAllCost();

//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<0,-1>(costRight);
	resetDirAccumulatedCostLast();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<-1,0>(costRight);
	resetDirAccumulatedCostLast();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;
/*
	aggregation<1,1>(costRight);
	resetDirAccumulatedCostLast();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;	
	aggregation<-1,1>(costRight);
	resetDirAccumulatedCostLast();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;
	aggregation<1,-1>(costRight);
	resetDirAccumulatedCostLast();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;
	aggregation<-1,-1>(costRight);
	resetDirAccumulatedCostLast();
	sumOverAllCost();
*/
	cv::Mat disparityTemp(HEIGHT, WIDTH, CV_8UC1);
	cv::Mat disparityLeftBackward(HEIGHT, WIDTH, CV_8UC1);
	createDisparity(disparityTemp);
	fastNlMeansDenoising(disparityTemp, disparityLeftBackward);
//	imshow("disparityLeftBackward", disparityLeftBackward);
	imwrite("./disparityLeftBackwardOLd.jpg", disparityLeftBackward);
//	consistencyCheck(disparityLeft, disparityLeftBackward, disparity, 0);


//	std::cout<<"consistency check"<<std::endl;
//	cv::Mat disparityLeftBackward =cv::imread("../disparityFlow00_backward.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//	cv::Mat disparityLeft =cv::imread("../disparityLeft.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//	consistencyCheck(disparityLeft, disparityLeftBackward, disparity);
	

}

void SGM::computeCost(){}

void SGM::computeCostRight(){}

SGM::~SGM(){
	censusImageRight.release();
	censusImageLeft.release();
	censusImageLeftLast.release();	
	cost.release();
	costRight.release();
	directCost.release();
	accumulatedCost.release();
}

void SGM::computeDerivative(){}

SGM::VecDf SGM::addPenalty(SGM::VecDf const& priorL,SGM::VecDf &localCost, float path_intensity_gradient ) {

	SGM::VecDf currL;
	float maxVal;

  	for ( int d = 0; d < DISP_RANGE; d++ ) {
    		float e_smooth = std::numeric_limits<float>::max();		
    		for ( int d_p = 0; d_p < DISP_RANGE; d_p++ ) {
      			if ( d_p - d == 0 ) {
        			// No penality
        			//e_smooth = std::min(e_smooth,priorL[d_p]);
				e_smooth = std::min(e_smooth,priorL[d]);
      			} else if ( abs(d_p - d) == 1 ) {
        			// Small penality
				e_smooth = std::min(e_smooth, priorL[d_p] + (PENALTY1));
      			} else {
        			// Large penality
				//maxVal=static_cast<float>(std::max((float)PENALTY1, path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2));
        			//maxVal=std::max(PENALTY1, path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2);				
				//e_smooth = std::min(e_smooth, (priorL[d_p] + maxVal));
				e_smooth = std::min(e_smooth, priorL[d_p] + PENALTY2);

      			}
    		}
    	currL[d] = localCost[d] + e_smooth;
  	}

	double minVal;
	cv::minMaxLoc(priorL, &minVal);

  	// Normalize by subtracting min of priorL cost
	for(int i = 0; i < DISP_RANGE; i++){
		currL[i] -= static_cast<float>(minVal);
	}

	return currL;
}

template <int DIRX, int DIRY>
void SGM::aggregation(cv::Mat cost) {

	if ( DIRX  == -1  && DIRY == 0) {
	std::cout<<"DIRECTION:(-1, 0) called,"<<std::endl;
      		// RIGHT MOST EDGE
      		for (int y = 0; y < HEIGHT; y++ ) {
			directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
      		}
      		for (int x = WIDTH - 2; x >= 0; x-- ) {
        		for ( int y = 0 ; y < HEIGHT ; y++ ) {
          			 directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
    	}	 

    	// Walk along the edges in a clockwise fashion
    	if ( DIRX == 1  && DIRY == 0) {
	std::cout<<"DIRECTION:( 1, 0) called,"<<std::endl;
      		// Process every pixel along this edge
     		for (int y = 0 ; y < HEIGHT ; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
     		for (int x = 1 ; x < WIDTH; x++ ) {
      			for ( int y = 0; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}

	if ( DIRX == 0 && DIRY == 1) {
	std::cout<<"DIRECTION:( 0, 1) called,"<<std::endl;
     		//TOP MOST EDGE	
      		for (int x = 0; x < WIDTH ; x++ ) {
			directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);	
      		}
      		for (int y = 1 ; y < HEIGHT ; y++ ) {   
			for ( int x = 0; x < WIDTH; x++ ) {
          			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
    	} 
	
	if ( DIRX == 0 && DIRY == -1) {
	std::cout<<"DIRECTION:( 0,-1) called,"<<std::endl;
      		// BOTTOM MOST EDGE
     		 for ( int x = 0 ; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
      		}
     		for ( int y = HEIGHT - 2; y >= 0; --y ) {
        		for ( int x = 0; x < WIDTH; x++ ) {
          			 directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
    	}

	if ( DIRX == 1  && DIRY == 1) {
	std::cout<<"DIRECTION:( 1, 1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = 1; x < WIDTH; x++ ) {
      			for ( int y = 1; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}

	if ( DIRX == 1  && DIRY == -1) {
	std::cout<<"DIRECTION:( 1,-1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = 1; x < WIDTH; x++ ) {
      			for ( int y = HEIGHT - 2; y >= 0 ; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}

	if ( DIRX == -1  && DIRY == 1) {
	std::cout<<"DIRECTION:(-1, 1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = WIDTH - 2; x >= 0; x-- ) {
      			for ( int y = 1; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}

	if ( DIRX == -1  && DIRY == -1) {
	std::cout<<"DIRECTION:(-1,-1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = WIDTH - 2; x >= 0; x-- ) {
      			for ( int y = HEIGHT - 2; y >= 0; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}

	if ( DIRX == 2  && DIRY == 1) {
	std::cout<<"DIRECTION:( 2, 1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = 2; x < WIDTH; x++ ) {
      			for ( int y = 2; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}

	if ( DIRX == 1  && DIRY == 2) {
	std::cout<<"DIRECTION:( 1, 2) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = 2; x < WIDTH; x++ ) {
      			for ( int y = 2; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}

	if ( DIRX == -2  && DIRY == 1) {
	std::cout<<"DIRECTION:(-2, 1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = WIDTH - 3; x >= 0; x-- ) {
      			for ( int y = 2; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}


	if ( DIRX == -1  && DIRY == 2) {
	std::cout<<"DIRECTION:(-1, 2) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = WIDTH - 3; x >= 0; x-- ) {
      			for ( int y = 2; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}

	if ( DIRX == 2  && DIRY == -1) {
	std::cout<<"DIRECTION:( 2,-1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = 2; x < WIDTH; x++ ) {
      			for ( int y = HEIGHT - 3; y >= 0 ; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}

	if ( DIRX == 1  && DIRY == -2) {
	std::cout<<"DIRECTION:( 1,-2) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = 2; x < WIDTH; x++ ) {
      			for ( int y = HEIGHT - 3; y >= 0 ; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}

	if ( DIRX == -2  && DIRY == -1) {
	std::cout<<"DIRECTION:(-2,-1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = WIDTH - 3; x >= 0; x-- ) {
      			for ( int y = HEIGHT - 3; y >= 0; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}

	if ( DIRX == -1  && DIRY == -2) {
	std::cout<<"DIRECTION:(-1,-2) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = WIDTH - 3; x >= 0; x-- ) {
      			for ( int y = HEIGHT - 3; y >= 0; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
        		}
      		}
	}
}



SGMFlow::SGMFlow(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_,
		 cv::Mat &EpipoleLeftLast_, cv::Mat &EpipoleLeft_, cv::Mat &fundamentalMatrix_)
		:SGM(imgLeftLast_, imgLeft_, imgRight_, PENALTY1_, PENALTY2_, winRadius_){
				
		EpipoleLeftLast_.copyTo(EpipoleLeftLast);
		EpipoleLeft_.copyTo(EpipoleLeft);
		fundamentalMatrix_.copyTo(fundamentalMatrix);
		imgRotation = Mat::zeros(HEIGHT, WIDTH, CV_32FC2);
		imgRotationBackward = Mat::zeros(HEIGHT, WIDTH, CV_32FC2);
		computeRotation();
		computeBackwardRotation();
		translationLeftLast = Mat::zeros(HEIGHT, WIDTH, CV_32FC2);
		translationLeft = Mat::zeros(HEIGHT, WIDTH, CV_32FC2);
		derivativeFlowLeftLast = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
		derivativeFlowLeft = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);

		disFlag = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
		disFlagBackward = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
}

SGMFlow::~SGMFlow(){
	imgRotation.release();
	imgRotationBackward.release();
	EpipoleLeftLast.release();
	EpipoleLeft.release();
	translationLeftLast.release();
	translationLeft.release();
	fundamentalMatrix.release();
	disFlag.release();
	disFlagBackward.release();
	derivativeFlowLeftLast.release();
	derivativeFlowLeft.release();
}

void SGMFlow::computeBackwardRotation(){
	//simply transpose the fundamental matrix

	cv::Mat fundamentalTranspose = fundamentalMatrix.t();
	const double cx = (double)WIDTH/2.0;
	const double cy = (double)HEIGHT/2.0;
	cv::Mat A(WIDTH*HEIGHT,5,CV_64FC1,cv::Scalar(0.0));
	cv::Mat b(WIDTH*HEIGHT,1,CV_64FC1,cv::Scalar(0.0));
	cv::Mat x_sol;
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			double x_, y_;
			cv::Mat hom(3,1,CV_64FC1);
			x_ = x - cx;
			y_ = y - cy;
			//generate homogenous coord
			hom.at<double>(0,0) = (double)x;
			hom.at<double>(1,0) = (double)y;
			hom.at<double>(2,0) = 1.0;
			//calc epiline through pixel
			cv::Mat epi = fundamentalTranspose*hom;
			A.at<double>(y*WIDTH+x,0) = epi.at<double>(0,0);
			A.at<double>(y*WIDTH+x,1) = epi.at<double>(1,0);
			A.at<double>(y*WIDTH+x,2) = (epi.at<double>(1,0)*x_)-(epi.at<double>(0,0)*y_);
			A.at<double>(y*WIDTH+x,3) = (epi.at<double>(0,0)*x_*x_)+(epi.at<double>(1,0)*x_*y_);
			A.at<double>(y*WIDTH+x,4) = (epi.at<double>(0,0)*x_*y_)+(epi.at<double>(1,0)*y_*y_);
			b.at<double>(y*WIDTH+x,0) = -epi.at<double>(2,0)-(epi.at<double>(0,0)*x)-(epi.at<double>(1,0)*y);
		}
	}
	cv::solve(A,b,x_sol,DECOMP_QR);
	
	std::cout<<"imgRotationBackward coef: "<<x_sol<<std::endl;
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			float x_, y_;
			x_ = x - cx;
			y_ = y - cy;
			
			imgRotationBackward.at<Vec2f>(y,x)[0] = (float)(x_sol.at<double>(0,0)-(x_sol.at<double>(2,0)*y_)+(x_sol.at<double>(3,0)*x_*x_)+(x_sol.at<double>(4,0)*x_*y_));
			imgRotationBackward.at<Vec2f>(y,x)[1] = (float)(x_sol.at<double>(1,0)+(x_sol.at<double>(2,0)*x_)+(x_sol.at<double>(3,0)*x_*y_)+(x_sol.at<double>(4,0)*y_*y_));
		}	
	}
	fundamentalTranspose.release();
	A.release();
	b.release();
	x_sol.release();
}

void SGMFlow::computeRotation(){

	const double cx = (double)WIDTH/2.0;
	const double cy = (double)HEIGHT/2.0;
	cv::Mat A(WIDTH*HEIGHT,5,CV_64FC1,cv::Scalar(0.0));
	cv::Mat b(WIDTH*HEIGHT,1,CV_64FC1,cv::Scalar(0.0));
	cv::Mat x_sol;
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			double x_, y_;
			cv::Mat hom(3,1,CV_64FC1);
			x_ = x - cx;
			y_ = y - cy;
			//generate homogenous coord
			hom.at<double>(0,0) = (double)x;
			hom.at<double>(1,0) = (double)y;
			hom.at<double>(2,0) = 1.0;
			//calc epiline through pixel
			cv::Mat epi = fundamentalMatrix*hom;
			A.at<double>(y*WIDTH+x,0) = epi.at<double>(0,0);
			A.at<double>(y*WIDTH+x,1) = epi.at<double>(1,0);
			A.at<double>(y*WIDTH+x,2) = (epi.at<double>(1,0)*x_)-(epi.at<double>(0,0)*y_);
			A.at<double>(y*WIDTH+x,3) = (epi.at<double>(0,0)*x_*x_)+(epi.at<double>(1,0)*x_*y_);
			A.at<double>(y*WIDTH+x,4) = (epi.at<double>(0,0)*x_*y_)+(epi.at<double>(1,0)*y_*y_);
			b.at<double>(y*WIDTH+x,0) = -epi.at<double>(2,0)-(epi.at<double>(0,0)*x)-(epi.at<double>(1,0)*y);
		}
	}
	cv::solve(A,b,x_sol,DECOMP_QR);
	
	std::cout<<"Rotation coef: "<<x_sol<<std::endl;
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			float x_, y_;
			x_ = x - cx;
			y_ = y - cy;
			
			imgRotation.at<Vec2f>(y,x)[0] = (float)(x_sol.at<double>(0,0)-(x_sol.at<double>(2,0)*y_)+(x_sol.at<double>(3,0)*x_*x_)+(x_sol.at<double>(4,0)*x_*y_));
			imgRotation.at<Vec2f>(y,x)[1] = (float)(x_sol.at<double>(1,0)+(x_sol.at<double>(2,0)*x_)+(x_sol.at<double>(3,0)*x_*y_)+(x_sol.at<double>(4,0)*y_*y_));
		}	
	}

	A.release();
	b.release();
	x_sol.release();
}


void SGMFlow::computeTranslation(cv::Mat &translation, cv::Mat &Epipole, float sign){
	
	for ( int x = 0; x < WIDTH; x++ ) {
      		for ( int y = 0; y < HEIGHT; y++ ) {
			float delta_x = sign*(x - Epipole.at<float>(0));
			float delta_y = sign*(y - Epipole.at<float>(1));
			float nomi = sqrt(delta_x*delta_x + delta_y*delta_y);
			float dir_x = (delta_x/nomi);
			float dir_y = (delta_y/nomi);
			translation.at<Vec2f>(y,x)[0]=dir_x;
			translation.at<Vec2f>(y,x)[1]=dir_y;
		}
	}

}


void SGMFlow::computeDerivative(){
float sobelCapValue_ = 15;
	cv::Mat gradxLeftLast, gradyLeftLast;
	cv::Sobel(imgLeftLast, gradxLeftLast, CV_32FC1, 1, 0);
	
	for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradxLeftLast.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			gradxLeftLast.at<float>(y,x) = sobelValue;
		}
	}
	

	cv::Sobel(imgLeftLast, gradyLeftLast, CV_32FC1, 0, 1);
	
	for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradyLeftLast.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			gradyLeftLast.at<float>(y,x) = sobelValue;
		}
	}
	
	cv::Mat gradxLeft, gradyLeft;
	cv::Sobel(imgLeft, gradxLeft, CV_32FC1, 1, 0);
	for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradxLeft.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			gradxLeft.at<float>(y,x) = sobelValue;
		}
	}
	
	cv::Sobel(imgLeft, gradyLeft, CV_32FC1, 0, 1);
	
	for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradyLeft.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			gradyLeft.at<float>(y,x) = sobelValue;
		}
	}
		
	computeTranslation(translationLeftLast, EpipoleLeftLast, 1.f);
	computeTranslation(translationLeft, EpipoleLeft, 1.f);


	for ( int x = 0; x < WIDTH; x++ ) {
      		for ( int y = 0; y < HEIGHT; y++ ) {

			derivativeFlowLeftLast.at<float>(y,x) = static_cast<float>(sqrt(pow(translationLeft.at<Vec2f>(y,x)[1]*gradyLeftLast.at<float>(y,x),2)+
										pow(translationLeft.at<Vec2f>(y,x)[0]*gradxLeftLast.at<float>(y,x),2)));

			derivativeFlowLeft.at<float>(y,x) = static_cast<float>(sqrt(pow(translationLeft.at<Vec2f>(y,x)[1]*gradyLeft.at<float>(y,x),2)+
										 pow(translationLeft.at<Vec2f>(y,x)[0]*gradxLeft.at<float>(y,x),2)));	
	
		}
	}

	gradxLeftLast.release();
	gradyLeftLast.release();
	gradxLeft.release();
	gradyLeft.release();


}


void SGMFlow::createDisparity(cv::Mat &disparity){
	
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			float imax = std::numeric_limits<float>::max();
			int min_index = 0;
			SGM::VecDf vec = accumulatedCost.at<SGM::VecDf>(y,x);

			for(int d = 0; d < DISP_RANGE; d++){
				if(vec[d] < imax ){ imax = vec[d]; min_index = d;}
			}
			disparity.at<uchar>(y,x) = static_cast<uchar>(DIS_FACTOR*min_index);	
		}
	}

}


void SGMFlow::consistencyCheck(cv::Mat disparityLeft, cv::Mat disparityLeftBackward, cv::Mat disparity, bool interpl){

	std::vector<Point2i> occFalse;
	cv::Mat disparityWoConsistency(HEIGHT, WIDTH, CV_8UC1);
	//based on backward
/*	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){
			disparityWoConsistency.at<uchar>(y,x) = static_cast<uchar>(disparityLeftBackward.at<uchar>(y,x) * visualDisparity);
			if(disFlag.at<uchar>(y,x) != static_cast<uchar>(DISFLAG)){
				unsigned short disparityLeftBackwardValue = static_cast<unsigned short>(disparityLeftBackward.at<uchar>(y,x));
				float newx = x+imgRotationBackward.at<Vec2f>(y,x)[0];
				float newy = y+imgRotationBackward.at<Vec2f>(y,x)[1];
				float distx = newx - EpipoleLeftLast.at<float>(0);
				float disty = newy - EpipoleLeftLast.at<float>(1);
						
				float L = sqrt(distx*distx + disty*disty);
				float d = L * ((float)Vmax*(float)disparityLeftBackwardValue/DISP_RANGE)/(1.0-((float)Vmax*(float)disparityLeftBackwardValue/DISP_RANGE));																
				int xx = round(newx - d*translationLeftLast.at<Vec2f>(newy,newx)[0]);
				int yy = round(newy - d*translationLeftLast.at<Vec2f>(newy,newx)[1]);

				unsigned short disparityLeftValue =  static_cast<unsigned short>(disparityLeft.at<uchar>(yy,xx));
				disparity.at<uchar>(y,x) = static_cast<uchar>(disparityLeftBackwardValue * visualDisparity);
				//disparityWoConsistency.at<uchar>(y,x) = static_cast<uchar>(disparityLeftBackwardValue * 3);
				if(abs(disparityLeftBackwardValue - disparityLeftValue) > disparityThreshold){
					disparity.at<uchar>(y,x) = static_cast<uchar>(Dinvd);
					occFalse.push_back(Point2i(x,y));
				}
			}else{
				unsigned short disparityLeftBackwardValue = static_cast<unsigned short>(disparityLeftBackward.at<uchar>(y,x));
				disparity.at<uchar>(y,x) = static_cast<uchar>(disparityLeftBackwardValue * visualDisparity);
			}		
		}
	}
*/	
	//based on forward
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){
			disparityWoConsistency.at<uchar>(y,x) = static_cast<uchar>(disparityLeftBackward.at<uchar>(y,x) * visualDisparity);
			unsigned short disparityLeftValue = static_cast<unsigned short>(disparityLeft.at<uchar>(y,x));
			float newx = x+imgRotation.at<Vec2f>(y,x)[0];
			float newy = y+imgRotation.at<Vec2f>(y,x)[1];
			float distx = newx - EpipoleLeft.at<float>(0);
			float disty = newy - EpipoleLeft.at<float>(1);
						
			float L = sqrt(distx*distx + disty*disty);
			float d = L * ((float)Vmax*(float)disparityLeftValue/DISP_RANGE)/(1.0-((float)Vmax*(float)disparityLeftValue/DISP_RANGE));																
			int xx = round(newx + d*translationLeft.at<Vec2f>(newy,newx)[0]);
			int yy = round(newy + d*translationLeft.at<Vec2f>(newy,newx)[1]);

			if((xx>=winRadius) && (yy>=winRadius) && xx<(WIDTH-winRadius) && yy< (HEIGHT-winRadius)){
				unsigned short disparityLeftBackwardValue =  static_cast<unsigned short>(disparityLeftBackward.at<uchar>(yy,xx));
				disparity.at<uchar>(y,x) = static_cast<uchar>(disparityLeftValue * visualDisparity);
				if(abs(disparityLeftBackwardValue - disparityLeftValue) > disparityThreshold){
					disparity.at<uchar>(y,x) = static_cast<uchar>(Dinvd);
					occFalse.push_back(Point2i(x,y));
				}
			}			
		}		
	}	
	


	const int len = occFalse.size();

	std::vector<unsigned short> newDisparity(len);

	for(int i = 0; i < len; i++){
		std::vector<int> neiborInfo;
		bool occlusion;
		int x = occFalse[i].x;
		int y = occFalse[i].y;
	
	if(interpl){

	
		{

			float xx = x + translationLeftLast.at<Vec2f>(y,x)[0];
			float yy = y + translationLeftLast.at<Vec2f>(y,x)[1];
			int dirx = 0;
			int dirxDisparityPx = 0;
			while((dirx <= 10) && (xx <= WIDTH - winRadius) && (yy <= HEIGHT - winRadius) && (xx >= winRadius) && (yy >= winRadius)){
				if(disparity.at<uchar>(round(yy),round(xx)) == static_cast<uchar>(Dinvd)) {
					xx = xx + translationLeftLast.at<Vec2f>(y,x)[0];
					yy = yy + translationLeftLast.at<Vec2f>(y,x)[1];
					continue;
				}
				dirxDisparityPx += static_cast<int>(disparity.at<uchar>(round(yy),round(xx)));
				xx = xx + translationLeftLast.at<Vec2f>(y,x)[0];
				yy = yy + translationLeftLast.at<Vec2f>(y,x)[1];
				dirx++;
			}
			if(dirx != 0){neiborInfo.push_back(round(dirxDisparityPx/(float)dirx));}
		}

		{
			float xx = x - translationLeftLast.at<Vec2f>(y,x)[0];
			float yy = y - translationLeftLast.at<Vec2f>(y,x)[1];
			int dirx = 0;
			int dirxDisparityNx = 0;
			while((dirx <= 10) && (xx <= WIDTH - winRadius) && (yy <= HEIGHT - winRadius) && (xx >= winRadius) && (yy >= winRadius)){
				if(disparity.at<uchar>(round(yy),round(xx)) == static_cast<uchar>(Dinvd)) {
					xx = xx - translationLeftLast.at<Vec2f>(y,x)[0];
					yy = yy - translationLeftLast.at<Vec2f>(y,x)[1];
					continue;
				}
				dirxDisparityNx += static_cast<int>(disparity.at<uchar>(round(yy),round(xx))); 
				xx = xx - translationLeftLast.at<Vec2f>(y,x)[0];
				yy = yy - translationLeftLast.at<Vec2f>(y,x)[1]; 
				dirx++;
			}
			if(dirx != 0){neiborInfo.push_back(round(dirxDisparityNx/(float)dirx));}
		}

		if(neiborInfo.size() == 2){ occlusion = fabs((neiborInfo[0]-neiborInfo[1])/fmin(neiborInfo[0], neiborInfo[1])) > 0.2 ? true : false;}
		else{occlusion = false;}

		{
			int yy = y + 1;
			int diry = 0;
			int dirxDisparityPy;
			while((diry < 1) && (yy <= HEIGHT - winRadius)){
				if(disparity.at<uchar>(yy,x) == static_cast<uchar>(Dinvd)) {yy++;continue;}
				dirxDisparityPy = static_cast<int>(disparity.at<uchar>(yy,x)); yy++; diry++;
			}
			if(diry != 0){neiborInfo.push_back(round(dirxDisparityPy/(float)diry));}
		}

		{
			int yy = y - 1;
			int diry = 0;
			int dirxDisparityNy;
			while((diry < 1) && (yy >= winRadius)){
				if(disparity.at<uchar>(yy,x) == static_cast<uchar>(Dinvd)) {yy--;continue;}
				dirxDisparityNy = static_cast<int>(disparity.at<uchar>(yy,x)); yy--; diry++;
			}
			if(diry != 0){neiborInfo.push_back(round(dirxDisparityNy/(float)diry));}

		}

		{
			int dirxy = 0;
			int yy = y + 1;
			int xx = x - 1;
			int dirxDisparityNxPy;
			while((dirxy < 1) && (yy <= HEIGHT - winRadius) && (xx >= winRadius)){
				if(disparity.at<uchar>(yy,xx) == static_cast<uchar>(Dinvd)) {yy++; xx--;continue;}
				dirxDisparityNxPy = static_cast<int>(disparity.at<uchar>(yy,xx)); yy++; xx--; dirxy++;
			}
			if(dirxy != 0){neiborInfo.push_back(round(dirxDisparityNxPy/(float)dirxy));}
		}

		{
			int dirxy = 0;
			int yy = y + 1;
			int xx = x + 1;
			int dirxDisparityPxPy;
			while((dirxy < 1) && (yy <= HEIGHT - winRadius) && (xx <= WIDTH - winRadius)){
				if(disparity.at<uchar>(yy,xx) == static_cast<uchar>(Dinvd)) {yy++; xx++;continue;}
				dirxDisparityPxPy = static_cast<int>(disparity.at<uchar>(yy,xx)); yy++; xx++; dirxy++;
			}
			if(dirxy != 0){neiborInfo.push_back(round(dirxDisparityPxPy/(float)dirxy));}
		}
		
		{
			int dirxy = 0;
			int yy = y - 1;
			int xx = x + 1;
			int dirxDisparityPxNy;
			while((dirxy < 1) && (yy >= winRadius) && (xx <= WIDTH - winRadius)){
				if(disparity.at<uchar>(yy,xx) == static_cast<uchar>(Dinvd)) {yy--; xx++;continue;}
				dirxDisparityPxNy = static_cast<int>(disparity.at<uchar>(yy,xx)); yy--; xx++; dirxy++;
			}
			if(dirxy != 0){neiborInfo.push_back(round(dirxDisparityPxNy/(float)dirxy));}
		}
		
		{
			int dirxy = 0;
			int yy = y - 1;
			int xx = x - 1;
			int dirxDisparityNxNy;
			while((dirxy < 1) && (yy >= winRadius) && (xx >= winRadius)){
				if(disparity.at<uchar>(yy,xx) == static_cast<uchar>(Dinvd)) {yy--; xx--;continue;}
				dirxDisparityNxNy = static_cast<int>(disparity.at<uchar>(yy,xx)); yy--; xx--; dirxy++;
			}
			if(dirxy != 0){neiborInfo.push_back(round(dirxDisparityNxNy/(float)dirxy));}
		}


		std::sort(neiborInfo.begin(), neiborInfo.end());



		int secLow = neiborInfo[2];
		int median = neiborInfo[floor(neiborInfo.size()/2.f)];


		unsigned short newValue = 0;	
		if(occlusion == true){
			newValue = secLow;			
		}else{
			newValue = median;	
		}
	
		newDisparity[i] = newValue ;

	}	

	for(int i = 0; i < len; i++){
		int x = occFalse[i].x;
		int y = occFalse[i].y;
		disparity.at<uchar>(y,x) = static_cast<uchar>(newDisparity[i]);

	}	
}

}


void SGMFlow::computeCostRight(){
//Scheme 2
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){

			for(int w = 0; w < DISP_RANGE ; w++){
				
				for(int neiY = y - winRadius ; neiY <= y + winRadius; neiY++){				
					for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
						float newx = neiX+imgRotationBackward.at<Vec2f>(neiY,neiX)[0];
						float newy = neiY+imgRotationBackward.at<Vec2f>(neiY,neiX)[1];
						float distx = newx - EpipoleLeftLast.at<float>(0);
						float disty = newy - EpipoleLeftLast.at<float>(1);
						//float distx = newx - EpipoleLeft.at<float>(0);
						//float disty = newy - EpipoleLeft.at<float>(1);

						float L = sqrt(distx*distx + disty*disty);
						//float d = L * ((float)Vmax*(float)w/DISP_RANGE);										
						float d = L * ((float)Vmax*(float)w/DISP_RANGE)/(1.0+((float)Vmax*(float)w/DISP_RANGE));										
						
						int xx = round(newx - d*translationLeftLast.at<Vec2f>(newy,newx)[0]);
						int yy = round(newy - d*translationLeftLast.at<Vec2f>(newy,newx)[1]);
				
						if((xx>=winRadius) && (yy>=winRadius) && xx<(WIDTH-winRadius) && yy< (HEIGHT-winRadius)){	
							
							costRight.at<SGM::VecDf>(y,x)[w] += fabs(derivativeFlowLeft.at<float>(neiY,neiX) - derivativeFlowLeftLast.at<float>(yy,xx))						
							+ (float)CENSUS_W * computeHammingDist(censusImageLeft.at<uchar>(neiY, neiX), censusImageLeftLast.at<uchar>(yy, xx));
							
										
						}else{
							//disFlagBackward.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
							costRight.at<SGM::VecDf>(y,x)[w] =100000; //costRight.at<SGM::VecDf>(y,x)[w-1];

						}
				
					}
				}
			}
		}
	}

//Set flag for image boundaries
	for(int y = 0; y < winRadius; y++){
		for(int x = 0; x < WIDTH; x++){
			disFlagBackward.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}
	for(int y = HEIGHT - 1; y > HEIGHT - 1 - winRadius; y--){
		for(int x = 0; x < WIDTH; x++){
			disFlagBackward.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}

	for(int x = 0; x < winRadius; x++){
		for(int y = 0; y < HEIGHT; y++){
			disFlagBackward.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}

	for(int x = WIDTH - 1; x > WIDTH -1 - winRadius; x--){
		for(int y = 0; y < HEIGHT; y++){
			disFlagBackward.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}


}


void SGMFlow::computeCost(){
	
//Scheme 2
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){

			for(int w = 0; w < DISP_RANGE ; w++){
				
				for(int neiY = y - winRadius ; neiY <= y + winRadius; neiY++){				
					for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
						float newx = neiX+imgRotation.at<Vec2f>(neiY,neiX)[0];
						float newy = neiY+imgRotation.at<Vec2f>(neiY,neiX)[1];
						float distx = newx - EpipoleLeft.at<float>(0);
						float disty = newy - EpipoleLeft.at<float>(1);
						

						float L = sqrt(distx*distx + disty*disty);
						float d = L * ((float)Vmax*(float)w/DISP_RANGE)/(1.0-((float)Vmax*(float)w/DISP_RANGE));										
						
						int xx = round(newx + d*translationLeft.at<Vec2f>(newy,newx)[0]);
						int yy = round(newy + d*translationLeft.at<Vec2f>(newy,newx)[1]);
				
						if((xx>=winRadius) && (yy>=winRadius) && xx<(WIDTH-winRadius) && yy< (HEIGHT-winRadius)){	
							
							cost.at<SGM::VecDf>(y,x)[w] += fabs(derivativeFlowLeftLast.at<float>(neiY,neiX) - derivativeFlowLeft.at<float>(yy,xx))						
							+ (float)CENSUS_W * computeHammingDist(censusImageLeftLast.at<uchar>(neiY, neiX), censusImageLeft.at<uchar>(yy, xx));
							
										
						}else{
							cost.at<SGM::VecDf>(y,x)[w] = cost.at<SGM::VecDf>(y,x)[w -1];
						}
				
					}
				}
			}
		}
	}
		
//Set flag for image boundaries
	for(int y = 0; y < winRadius; y++){
		for(int x = 0; x < WIDTH; x++){
			disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}
	for(int y = HEIGHT - 1; y > HEIGHT - 1 - winRadius; y--){
		for(int x = 0; x < WIDTH; x++){
			disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}

	for(int x = 0; x < winRadius; x++){
		for(int y = 0; y < HEIGHT; y++){
			disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}

	for(int x = WIDTH - 1; x > WIDTH -1 - winRadius; x--){
		for(int y = 0; y < HEIGHT; y++){
			disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}


}

void SGMFlow::postProcess(cv::Mat &disparity){

	for(int x = 0; x < WIDTH; x++){
		for(int y = 0; y < HEIGHT; y++){
			if(disFlag.at<uchar>(y,x) == static_cast<uchar>(DISFLAG)){disparity.at<uchar>(y,x)=static_cast<uchar>(0);}
		}
	}

}

void SGMFlow::writeDerivative(){
	imwrite("../derivativeFlowLeft.jpg",derivativeFlowLeft);
	imwrite("../derivativeFlowLeftLast.jpg",derivativeFlowLeftLast);
}


void SGMFlow::copyDisflag(cv::Mat &M){

	disFlag.copyTo(M);

}

void SGMFlow::computeFlow(cv::Mat &disparity, std::string fileName){

	float *out = (float *)calloc(HEIGHT * WIDTH *3, sizeof(float));

	for(int x = winRadius; x < WIDTH - winRadius; x++){
		for(int y = winRadius; y < HEIGHT - winRadius; y++){
			if(disparity.at<uchar>(y,x) == static_cast<uchar>(Dinvd)){
				out[(WIDTH*y*3)+(x*3)+2] = static_cast<float>(0);
			
			}else{
				unsigned short disparityLeftValue = static_cast<unsigned short>(disparity.at<uchar>(y,x));
				float newx = x+imgRotation.at<Vec2f>(y,x)[0];
				float newy = y+imgRotation.at<Vec2f>(y,x)[1];
				float distx = newx - EpipoleLeft.at<float>(0);
				float disty = newy - EpipoleLeft.at<float>(1);
						
				float L = sqrt(distx*distx + disty*disty);
				float d = L * ((float)Vmax*(float)disparityLeftValue/DISP_RANGE)/(1.0-((float)Vmax*(float)disparityLeftValue/DISP_RANGE));																
				float Vx = d*translationLeft.at<Vec2f>(newy,newx)[0]+imgRotation.at<Vec2f>(y,x)[0];
				float Vy = d*translationLeft.at<Vec2f>(newy,newx)[1]+imgRotation.at<Vec2f>(y,x)[1];

				out[(WIDTH*y*3)+(x*3)+0] = Vx;
				out[(WIDTH*y*3)+(x*3)+1] = Vy;
				out[(WIDTH*y*3)+(x*3)+2] = static_cast<float>(1);
			}


		}
	}

	std::string fileColorName = "color" + fileName;
	FlowImage fi(out, WIDTH, HEIGHT);
	fi.write(fileName);
//	fi.writeColor(fileColorName);
	free(out);
}




