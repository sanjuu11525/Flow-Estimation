#include "SGM.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/stitching/detail/matchers.hpp"

void computeEpipoles(std::vector<cv::Vec3f> &lines, cv::Mat &x_sol);

int main(int argc, char *argv[]){

	 cv::Mat image_left, image_left_last;
	 cv::Mat gray_left, gray_right, gray_left_last;
	 gray_left  = cv::imread("",CV_LOAD_IMAGE_GRAYSCALE);
 	gray_right = cv::imread("",CV_LOAD_IMAGE_GRAYSCALE);
	 gray_left_last  = cv::imread("",CV_LOAD_IMAGE_GRAYSCALE);
	 image_left_last = cv::imread("",CV_LOAD_IMAGE_COLOR);
	 image_left      = cv::imread("",CV_LOAD_IMAGE_COLOR);

	 const cv::Scalar colorBlue(225.0, 0.0, 0.0, 0.0);
	 const cv::Scalar colorRed(0.0, 0.0, 225.0, 0.0);

  sgmflow::SGMStereoParameters parameter;
  parameter.width  = image_left.cols;
  parameter.height = image_left.rows;

//-- compute the flow part
	 const int HEIGHT = gray_left.rows;
	 const int WIDTH = gray_left.cols;

	 std::vector<cv::DMatch> matches1to2;
	 std::vector<cv::KeyPoint> keypoints_1;
	 std::vector<cv::KeyPoint> keypoints_2;

	 cv::Mat descriptors_1, descriptors_2;
	 cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
	 sift->detectAndCompute(gray_left_last,cv::noArray(),keypoints_1,descriptors_1);
	 sift->detectAndCompute(gray_left,cv::noArray(),keypoints_2,descriptors_2);
	 cv:: FlannBasedMatcher matcher;
	 matcher.match(descriptors_1, descriptors_2, matches1to2);

	 double max_dist = 0; double min_dist = 100;
  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < matches1to2.size(); i++ ){
		  double dist = matches1to2[i].distance;
    		if( dist < min_dist ) min_dist = dist;
    		if( dist > max_dist ) max_dist = dist;
  }

  //-- Draw only "good" matches (i.e. whose distance is less than C*min_dist )
 	std::vector<cv::DMatch > good_matches;
	 std::vector<cv::Point2f> temp_keypoints_1;
	 std::vector<cv::Point2f> temp_keypoints_2;

 	for( int i = 0; i < matches1to2.size(); i++ ){ 
		  if( matches1to2[i].distance <= 16*min_dist ){
			   good_matches.push_back( matches1to2[i]);
			   temp_keypoints_1.push_back((cv::Point2f)keypoints_1[matches1to2[i].queryIdx].pt);
			   temp_keypoints_2.push_back((cv::Point2f)keypoints_2[matches1to2[i].trainIdx].pt);
		  }
  }

	 cv::Mat mask;
	 cv::Mat Fmat;
	 Fmat=findFundamentalMat(temp_keypoints_1, temp_keypoints_2, CV_FM_RANSAC , 3, 0.99 ,mask); //CV_FM_8POINT, CV_FM_RANSAC, CV_FM_LMEDS

	 std::vector<cv::Point2f> new_keypoints_1;
	 std::vector<cv::Point2f> new_keypoints_2;
	 //filtering out non-qualified matchings by RANSAC
	 int num_inliers = 0;
	 for (int i = 1; i < temp_keypoints_1.size(); i++){
		  cv::Point_<uchar> pp= mask.at<cv::Point_<uchar> >(i,0);
		  cv::Point2f a=temp_keypoints_1[i];
		  cv::Point2f b=temp_keypoints_1[i-1];
		  if((short)pp.x != 0 && (a.x != b.x) && (b.y != a.y)){
			   new_keypoints_1.push_back(temp_keypoints_1[i]);
			   new_keypoints_2.push_back(temp_keypoints_2[i]);
			   num_inliers++;
		  }
 	}

	//-----------------------------------------------------------------------------------
	//updated 23.11.2017. The fundamental matrix must be recomputed again with CV_FM_8POINT by qualified matchings.
	cv::Mat newFmat = findFundamentalMat(new_keypoints_1, new_keypoints_2, CV_FM_8POINT);
	std::cout<<"newFmat"<<newFmat<<std::endl;

	std::cout<<"num_inliers : "<<num_inliers<<std::endl;
	std::cout<<new_keypoints_1.size()<<std::endl;
	std::cout<<temp_keypoints_1.size()<<std::endl;

	std::vector<cv::Vec3f> lines_1;
	std::vector<cv::Vec3f> lines_2;
	cv::computeCorrespondEpilines(new_keypoints_2, 2, newFmat, lines_1);
	cv::computeCorrespondEpilines(new_keypoints_1, 1, newFmat, lines_2);

	std::vector<cv::Point2f> des_points_1;
	std::vector<cv::Point2f> des_points_2;

  //-- Compute epipoles by least square
	 cv::Mat Epipole_1 = cv::Mat::zeros(2, 1, CV_32F);
	 computeEpipoles(lines_1, Epipole_1);

	 cv::Mat Epipole_2 = cv::Mat::zeros(2, 1, CV_32F);
	 computeEpipoles(lines_2, Epipole_2);

  //-- draw epipolar lines on image
	 des_points_2.push_back(cv::Point2f(Epipole_2.at<float>(0), Epipole_2.at<float>(1)));
	 des_points_1.push_back(cv::Point2f(Epipole_1.at<float>(0), Epipole_1.at<float>(1)));

	 for (int i=0; i < new_keypoints_1.size(); i++){
		  cv::line(image_left_last, new_keypoints_1[i], des_points_1[0], colorRed,1,8);
		  cv::line(image_left_last, new_keypoints_2[i], des_points_2[0], colorBlue,1,8);
		  cv::line(image_left, new_keypoints_2[i], des_points_2[0], colorBlue,1,8);
		  cv::circle(image_left_last, new_keypoints_1[i],10,colorRed,1);
		  cv::circle(image_left, new_keypoints_2[i],10,colorBlue,1);
	 }

	cv::Mat disparityFlow_(gray_left.rows, gray_left.cols, CV_8UC1);
	sgmflow::SGMFlow sgmflow(gray_left_last, gray_left, gray_right, parameter, Epipole_1, Epipole_2, newFmat);
	sgmflow.process(disparityFlow_);
	sgmflow.computeFlow(disparityFlow_, "disparity.png");

	cv::waitKey(0);
	
	return 0;
}

void computeEpipoles(std::vector<cv::Vec3f> &lines, cv::Mat &x_sol){

	const int cols = 2;
	const int rows = lines.size();	
	
	cv::Mat A = cv::Mat::zeros(rows, cols, CV_32FC1);
	cv::Mat b = cv::Mat::zeros(rows, 1, CV_32FC1);

	for(int i = 0; i < rows; i++){
		A.at<float>(i,0) = lines[i][0];
		A.at<float>(i,1) = lines[i][1];
		b.at<float>(i,0) = -1.0 * lines[i][2];
	}
	cv::solve(A,b,x_sol,cv::DECOMP_QR);
}



