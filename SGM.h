/**
 * \brief This file defines a simple SGM example based on oOpenCV library.
 * \author      SanYu Huang (sanjuu11525@hotmail.com)
 * \maintainer  SanYu Huang (sanjuu11525@hotmail.com)
 * \copyright   There is no copy right. The file is recommended for education purpose.
 */
#pragma once

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <limits>
#include <iostream>
#define DISP_RANGE 130
#define DIS_FACTOR 1
#define CENSUS_W 5
#define DISFLAG 255
#define Vmax 0.7
#define disparityThreshold 2
#define Dinvd 0
#define visualDisparity 1


namespace sgmflow {

struct SGMStereoParameters {
  int penalty1 = 400;
  int penalty2 = 6000;
  int win_radius = 2;
  float sobel_cap_value = 15;
  int height;
  int width;
};

class SGM {
public:
  typedef cv::Vec<float, DISP_RANGE> vector_type;
  typedef SGMStereoParameters parameter_type;

protected:
  const cv::Mat &img_left_;
  const cv::Mat &img_right_;
  const cv::Mat &img_left_last_;
  cv::Mat census_image_right_;
  cv::Mat census_image_left_;
  cv::Mat census_image_left_last_;
  cv::Mat cost_left_to_right_;
  cv::Mat cost_right_to_left_;
  cv::Mat accumulated_cost_;

  void computeCensus(const cv::Mat &image, cv::Mat &censusImg);
  int  computeHammingDist(const uchar left, const uchar right);
  vector_type addPenalty(vector_type const& prior, vector_type &local);
  void sumOverAllCost(cv::Mat& pathWiseCost);
  virtual void createDisparity(cv::Mat &disparity);
  template <int DIRX, int DIRY> void aggregation(cv::Mat cost);
  virtual void computeDerivative() = 0;
  virtual void computeCost() = 0;
  virtual void computeCostRight() = 0;
  virtual void consistencyCheck(cv::Mat disparityLeft, cv::Mat disparityRight, cv::Mat disparity, bool interpl) = 0;

//  void aggregationCUDA(cv::Mat &directAcc, cv::Mat &cost);

public:
  SGM(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, parameter_type parameters);
  void process(cv::Mat &disparity);
  virtual ~SGM();

protected:
  parameter_type parameters_;
};


class SGMFlow : public SGM {
protected:
  cv::Mat img_rotation_;
  cv::Mat img_rotation_backward_;
  cv::Mat epipole_left_last_;
  cv::Mat epipole_left_;
  cv::Mat translation_left_last_;
  cv::Mat translation_left_;
  cv::Mat fundamental_matrix_;
  cv::Mat derivative_flow_left_last_;
  cv::Mat derivative_flow_left_;

  virtual void createDisparity(cv::Mat &disparity);
  virtual void computeDerivative();
  virtual void computeCost();
  virtual void computeCostRight();
  void computeRotation();
  void computeBackwardRotation();
  void computeTranslation(cv::Mat &translation, cv::Mat &Epipole, float sign);
  virtual void consistencyCheck(cv::Mat disparityLeft, cv::Mat disparityRight, cv::Mat disparity, bool interpl);

public:
  SGMFlow(const cv::Mat &imgLeftLast, const cv::Mat &imgLeft, const cv::Mat &imgRight, parameter_type& parameters,
        cv::Mat &EpipoleLeftLast_, cv::Mat &EpipoleLeft_, cv::Mat &fundamentalMatrix_);
  virtual ~SGMFlow();
  void computeFlow(cv::Mat &disparity, std::string fileName);
};
}