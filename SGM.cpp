#include "SGM.h"
// io_flow.h is from KITTI development kit
#include "io_flow.h"
#include "opencv2/photo.hpp"

namespace sgmflow {
SGM::SGM(const cv::Mat &imgLeftLast, const cv::Mat &img_left_, const cv::Mat &imgRight, parameter_type parameters)
: img_left_last_(imgLeftLast), img_left_(img_left_), img_right_(imgRight), parameters_(parameters) {

  const int width  = parameters_.width;
  const int height = parameters_.height;

  census_image_left_      = cv::Mat::zeros(height, width, CV_8UC1);
  census_image_right_     = cv::Mat::zeros(height, width, CV_8UC1);
  census_image_left_last_ = cv::Mat::zeros(height, width, CV_8UC1);
  cost_left_to_right_  = cv::Mat::zeros(height, width, CV_32FC(DISP_RANGE));
  cost_right_to_left_  = cv::Mat::zeros(height, width, CV_32FC(DISP_RANGE));
  accumulated_cost_    = cv::Mat::zeros(height, width, CV_32SC(DISP_RANGE));
};

void SGM::computeCensus(const cv::Mat &image, cv::Mat &censusImg) {
  const int width  = parameters_.width;
  const int height = parameters_.height;
  const int win_radius = parameters_.win_radius;

  for (int y = win_radius + 1; y < height - win_radius - 1; ++y) {
    for (int x = win_radius; x < width - win_radius; ++x) {
      unsigned char centerValue = image.at<uchar>(y, x);
      int censusCode = 0;
      for (int neiY = -win_radius - 1; neiY <= win_radius + 1; ++neiY) {
        for (int neiX = -win_radius; neiX <= win_radius; ++neiX) {
          censusCode = censusCode << 1;
          if (image.at<uchar>(y + neiY, x + neiX) >= centerValue) censusCode += 1;
        }
      }
      censusImg.at<uchar>(y, x) = static_cast<unsigned char>(censusCode);
    }
  }
};

int SGM::computeHammingDist(const uchar left, const uchar right) {
  int var = static_cast<int>(left ^ right);
  int count = 0;
  while (var) {
    var = var & (var - 1);
    count++;
  }
  return count;
};

void SGM::sumOverAllCost(cv::Mat& pathWiseCost) {

  const int width  = parameters_.width;
  const int height = parameters_.height;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      accumulated_cost_.at<vector_type>(y, x) += pathWiseCost.at<vector_type>(y, x);
    }
  }
};

void SGM::createDisparity(cv::Mat& disparity) {

  const int width  = parameters_.width;
  const int height = parameters_.height;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float imax = std::numeric_limits<float>::max();
      int min_index = 0;
      vector_type vec = accumulated_cost_.at<vector_type>(y, x);

      for (int d = 0; d < DISP_RANGE; d++) {
        if (vec[d] < imax) {
          imax = vec[d];
          min_index = d;
        }
      }
      disparity.at<uchar>(y, x) = static_cast<uchar>(DIS_FACTOR * min_index);
    }
  }
};

void SGM::process(cv::Mat &disparity) {

  const int width  = parameters_.width;
  const int height = parameters_.height;
  computeCensus(img_left_    , census_image_left_);
  computeCensus(img_right_   , census_image_right_);
  computeCensus(img_left_last_, census_image_left_last_);
  computeDerivative();
  computeCost();

  aggregation<1, 0>(cost_left_to_right_);
  aggregation<0, 1>(cost_left_to_right_);
  aggregation<-1, 0>(cost_left_to_right_);
  aggregation<0, -1>(cost_left_to_right_);

  cv::Mat disparity_left(height, width, CV_8UC1);
  cv::Mat disparity_temp(height, width, CV_8UC1);
  createDisparity(disparity_temp);
  fastNlMeansDenoising(disparity_temp, disparity_left);

  accumulated_cost_.setTo(0.0);

  computeCostRight();

  aggregation<1, 0>(cost_right_to_left_);
  aggregation<0, 1>(cost_right_to_left_);
  aggregation<0, -1>(cost_right_to_left_);
  aggregation<-1, 0>(cost_right_to_left_);

  cv::Mat disparity_right(height, width, CV_8UC1);
  createDisparity(disparity_temp);
  fastNlMeansDenoising(disparity_temp, disparity_right);

  consistencyCheck(disparity_left, disparity_right, disparity, 0);
};

SGM::~SGM() {
  census_image_right_.release();
  census_image_left_.release();
  census_image_left_last_.release();
  cost_left_to_right_.release();
  cost_right_to_left_.release();
  accumulated_cost_.release();
};

SGM::vector_type SGM::addPenalty(vector_type const &priorL, vector_type &local_cost) {

  const int penalty1 = parameters_.penalty1;
  const int penalty2 = parameters_.penalty2;

  vector_type penalized_disparity;

  for (int d = 0; d < DISP_RANGE; d++) {
    float e_smooth = std::numeric_limits<float>::max();
    for (int d_p = 0; d_p < DISP_RANGE; d_p++) {
      if (d_p - d == 0) {
        //e_smooth = std::min(e_smooth,priorL[d_p]);
        e_smooth = std::min(e_smooth, priorL[d]);
      } else if (abs(d_p - d) == 1) {
        // Small penality
        e_smooth = std::min(e_smooth, priorL[d_p] + penalty1);
      } else {
        // Large penality
        e_smooth = std::min(e_smooth, priorL[d_p] + penalty2);
      }
    }
    penalized_disparity[d] = local_cost[d] + e_smooth;
  }

  double minVal;
  cv::minMaxLoc(priorL, &minVal);

  // Normalize by subtracting min of priorL cost_
  for (int i = 0; i < DISP_RANGE; i++) {
    penalized_disparity[i] -= static_cast<float>(minVal);
  }

  return penalized_disparity;
};

template<int DIRX, int DIRY>
void SGM::aggregation(cv::Mat cost) {

  const int width  = parameters_.width;
  const int height = parameters_.height;
  cv::Mat pathWiseCost = cv::Mat::zeros(height, width, CV_32SC(DISP_RANGE));

  if (DIRX == -1 && DIRY == 0) {
    // RIGHT MOST EDGE
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = width - 2; x >= 0; --x) {
      for (int y = 0; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  // Walk along the edges in a clockwise fashion
  if (DIRX == 1 && DIRY == 0) {
    // Process every pixel along this edge
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 1; x < width; ++x) {
      for (int y = 0; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 0 && DIRY == 1) {
    //TOP MOST EDGE
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }
    for (int y = 1; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 0 && DIRY == -1) {
    // BOTTOM MOST EDGE
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }
    for (int y = height - 2; y >= 0; --y) {
      for (int x = 0; x < width; ++x) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 1 && DIRY == 1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }

    for (int x = 1; x < width; ++x) {
      for (int y = 1; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 1 && DIRY == -1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }

    for (int x = 1; x < width; ++x) {
      for (int y = height - 2; y >= 0; --y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == -1 && DIRY == 1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }

    for (int x = width - 2; x >= 0; --x) {
      for (int y = 1; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == -1 && DIRY == -1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }

    for (int x = width - 2; x >= 0; --x) {
      for (int y = height - 2; y >= 0; --y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 2 && DIRY == 1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }

    for (int x = 2; x < width; ++x) {
      for (int y = 2; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 1 && DIRY == 2) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }

    for (int x = 2; x < width; ++x) {
      for (int y = 2; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == -2 && DIRY == 1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }

    for (int x = width - 3; x >= 0; --x) {
      for (int y = 2; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }


  if (DIRX == -1 && DIRY == 2) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }

    for (int x = width - 3; x >= 0; --x) {
      for (int y = 2; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 2 && DIRY == -1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }

    for (int x = 2; x < width; ++x) {
      for (int y = height - 3; y >= 0; --y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 1 && DIRY == -2) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }

    for (int x = 2; x < width; ++x) {
      for (int y = height - 3; y >= 0; --y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == -2 && DIRY == -1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }

    for (int x = width - 3; x >= 0; --x) {
      for (int y = height - 3; y >= 0; --y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == -1 && DIRY == -2) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }

    for (int x = width - 3; x >= 0; --x) {
      for (int y = height - 3; y >= 0; --y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                        cost.at<vector_type>(y, x));
      }
    }
  }

  sumOverAllCost(pathWiseCost);
};

SGMFlow::SGMFlow(const cv::Mat &imgLeftLast, const cv::Mat &imgLeft, const cv::Mat &imgRight, parameter_type& parameters,
cv::Mat &epipole_left_last, cv::Mat &epipole_left, cv::Mat &fundamental_matrix)
: SGM(imgLeftLast, imgLeft, imgRight, parameters) {

  const int width  = parameters_.width;
  const int height = parameters_.height;

  img_rotation_          = cv::Mat::zeros(height, width, CV_32FC2);
  img_rotation_backward_ = cv::Mat::zeros(height, width, CV_32FC2);
  translation_left_last_ = cv::Mat::zeros(height, width, CV_32FC2);
  translation_left_      = cv::Mat::zeros(height, width, CV_32FC2);
  derivative_flow_left_  = cv::Mat::zeros(height, width, CV_32FC1);
  derivative_flow_left_last_ = cv::Mat::zeros(height, width, CV_32FC1);

  epipole_left_last.copyTo(epipole_left_last_);
  epipole_left.copyTo(epipole_left_);
  fundamental_matrix.copyTo(fundamental_matrix_);
  computeRotation();
  computeBackwardRotation();
};

SGMFlow::~SGMFlow() {
  img_rotation_.release();
  img_rotation_backward_.release();
  epipole_left_last_.release();
  epipole_left_.release();
  translation_left_last_.release();
  translation_left_.release();
  fundamental_matrix_.release();
  derivative_flow_left_last_.release();
  derivative_flow_left_.release();
};

void SGMFlow::computeBackwardRotation() {

  const int width  = parameters_.width;
  const int height = parameters_.height;

  cv::Mat fundamentalTranspose = fundamental_matrix_.t();
  const double cx = (double) width / 2.0;
  const double cy = (double) height / 2.0;
  cv::Mat A(width *height,
  5, CV_64FC1, cv::Scalar(0.0));
  cv::Mat b(width *height,
  1, CV_64FC1, cv::Scalar(0.0));
  cv::Mat x_sol;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double x_, y_;
      cv::Mat hom(3, 1, CV_64FC1);
      x_ = x - cx;
      y_ = y - cy;
      //generate homogenous coord
      hom.at<double>(0, 0) = (double) x;
      hom.at<double>(1, 0) = (double) y;
      hom.at<double>(2, 0) = 1.0;
      //calc epiline through pixel
      cv::Mat epi = fundamentalTranspose * hom;
      A.at<double>(y * width + x, 0) = epi.at<double>(0, 0);
      A.at<double>(y * width + x, 1) = epi.at<double>(1, 0);
      A.at<double>(y * width + x, 2) = (epi.at<double>(1, 0) * x_) - (epi.at<double>(0, 0) * y_);
      A.at<double>(y * width + x, 3) = (epi.at<double>(0, 0) * x_ * x_) + (epi.at<double>(1, 0) * x_ * y_);
      A.at<double>(y * width + x, 4) = (epi.at<double>(0, 0) * x_ * y_) + (epi.at<double>(1, 0) * y_ * y_);
      b.at<double>(y * width + x, 0) = -epi.at<double>(2, 0) - (epi.at<double>(0, 0) * x) - (epi.at<double>(1, 0) * y);
    }
  }
  cv::solve(A, b, x_sol, cv::DECOMP_QR);

  std::cout << "img_rotation_backward_ coef: " << x_sol << std::endl;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float x_, y_;
      x_ = x - cx;
      y_ = y - cy;

      img_rotation_backward_.at<vector_type>(y, x)[0] = (float) (x_sol.at<double>(0, 0) - (x_sol.at<double>(2, 0) * y_) +
                                                        (x_sol.at<double>(3, 0) * x_ * x_) +
                                                        (x_sol.at<double>(4, 0) * x_ * y_));
      img_rotation_backward_.at<vector_type>(y, x)[1] = (float) (x_sol.at<double>(1, 0) + (x_sol.at<double>(2, 0) * x_) +
                                                        (x_sol.at<double>(3, 0) * x_ * y_) +
                                                        (x_sol.at<double>(4, 0) * y_ * y_));
    }
  }
  fundamentalTranspose.release();
  A.release();
  b.release();
  x_sol.release();
};

void SGMFlow::computeRotation() {

  const int width  = parameters_.width;
  const int height = parameters_.height;
  
  const double cx = (double) width / 2.0;
  const double cy = (double) height / 2.0;
  cv::Mat A(width *height,
  5, CV_64FC1, cv::Scalar(0.0));
  cv::Mat b(width *height,
  1, CV_64FC1, cv::Scalar(0.0));
  cv::Mat x_sol;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double x_, y_;
      cv::Mat hom(3, 1, CV_64FC1);
      x_ = x - cx;
      y_ = y - cy;
      //generate homogenous coord
      hom.at<double>(0, 0) = (double) x;
      hom.at<double>(1, 0) = (double) y;
      hom.at<double>(2, 0) = 1.0;
      //calc epiline through pixel
      cv::Mat epi = fundamental_matrix_ * hom;
      A.at<double>(y * width + x, 0) = epi.at<double>(0, 0);
      A.at<double>(y * width + x, 1) = epi.at<double>(1, 0);
      A.at<double>(y * width + x, 2) = (epi.at<double>(1, 0) * x_) - (epi.at<double>(0, 0) * y_);
      A.at<double>(y * width + x, 3) = (epi.at<double>(0, 0) * x_ * x_) + (epi.at<double>(1, 0) * x_ * y_);
      A.at<double>(y * width + x, 4) = (epi.at<double>(0, 0) * x_ * y_) + (epi.at<double>(1, 0) * y_ * y_);
      b.at<double>(y * width + x, 0) = -epi.at<double>(2, 0) - (epi.at<double>(0, 0) * x) - (epi.at<double>(1, 0) * y);
    }
  }
  cv::solve(A, b, x_sol, cv::DECOMP_QR);

  std::cout << "Rotation coef: " << x_sol << std::endl;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float x_, y_;
      x_ = x - cx;
      y_ = y - cy;

      img_rotation_.at<vector_type>(y, x)[0] = (float) (x_sol.at<double>(0, 0) - (x_sol.at<double>(2, 0) * y_) +
                                                (x_sol.at<double>(3, 0) * x_ * x_) +
                                                (x_sol.at<double>(4, 0) * x_ * y_));
      img_rotation_.at<vector_type>(y, x)[1] = (float) (x_sol.at<double>(1, 0) + (x_sol.at<double>(2, 0) * x_) +
                                                (x_sol.at<double>(3, 0) * x_ * y_) +
                                                (x_sol.at<double>(4, 0) * y_ * y_));
    }
  }

  A.release();
  b.release();
  x_sol.release();
};

void SGMFlow::computeTranslation(cv::Mat &translation, cv::Mat &Epipole, float sign) {

  const int width  = parameters_.width;
  const int height = parameters_.height;
  
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      float delta_x = sign * (x - Epipole.at<float>(0));
      float delta_y = sign * (y - Epipole.at<float>(1));
      float nomi = sqrt(delta_x * delta_x + delta_y * delta_y);
      float dir_x = (delta_x / nomi);
      float dir_y = (delta_y / nomi);
      translation.at<vector_type>(y, x)[0] = dir_x;
      translation.at<vector_type>(y, x)[1] = dir_y;
    }
  }
};

void SGMFlow::computeDerivative() {

  const int width = parameters_.width;
  const int height = parameters_.height;
  const float sobel_cap_value = parameters_.sobel_cap_value;

  cv::Mat gradx_left_last, grady_left_last;
  cv::Sobel(img_left_last_, gradx_left_last, CV_32FC1, 1, 0);

  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      float sobelValue = gradx_left_last.at<float>(y, x);
      if (sobelValue > sobel_cap_value) sobelValue = 2 * sobel_cap_value;
      else if (sobelValue < -sobel_cap_value) sobelValue = 0;
      else sobelValue += sobel_cap_value;
      gradx_left_last.at<float>(y, x) = sobelValue;
    }
  }


  cv::Sobel(img_left_last_, grady_left_last, CV_32FC1, 0, 1);

  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      float sobelValue = grady_left_last.at<float>(y, x);
      if (sobelValue > sobel_cap_value) sobelValue = 2 * sobel_cap_value;
      else if (sobelValue < -sobel_cap_value) sobelValue = 0;
      else sobelValue += sobel_cap_value;
      grady_left_last.at<float>(y, x) = sobelValue;
    }
  }

  cv::Mat gradxLeft, gradyLeft;
  cv::Sobel(img_left_, gradxLeft, CV_32FC1, 1, 0);
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      float sobelValue = gradxLeft.at<float>(y, x);
      if (sobelValue > sobel_cap_value) sobelValue = 2 * sobel_cap_value;
      else if (sobelValue < -sobel_cap_value) sobelValue = 0;
      else sobelValue += sobel_cap_value;
      gradxLeft.at<float>(y, x) = sobelValue;
    }
  }

  cv::Sobel(img_left_, gradyLeft, CV_32FC1, 0, 1);

  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      float sobelValue = gradyLeft.at<float>(y, x);
      if (sobelValue > sobel_cap_value) sobelValue = 2 * sobel_cap_value;
      else if (sobelValue < -sobel_cap_value) sobelValue = 0;
      else sobelValue += sobel_cap_value;
      gradyLeft.at<float>(y, x) = sobelValue;
    }
  }

  computeTranslation(translation_left_last_, epipole_left_last_, 1.f);
  computeTranslation(translation_left_, epipole_left_, 1.f);
  
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      derivative_flow_left_last_.at<float>(y, x) = static_cast<float>(sqrt(
      pow(translation_left_.at<vector_type>(y, x)[1] * grady_left_last.at<float>(y, x), 2) +
      pow(translation_left_.at<vector_type>(y, x)[0] * gradx_left_last.at<float>(y, x), 2)));
      derivative_flow_left_.at<float>(y, x) = static_cast<float>(sqrt(
      pow(translation_left_.at<vector_type>(y, x)[1] * gradyLeft.at<float>(y, x), 2) +
      pow(translation_left_.at<vector_type>(y, x)[0] * gradxLeft.at<float>(y, x), 2)));
    }
  }

  gradx_left_last.release();
  grady_left_last.release();
  gradxLeft.release();
  gradyLeft.release();
};

void SGMFlow::createDisparity(cv::Mat &disparity) {

  const int width  = parameters_.width;
  const int height = parameters_.height;
  
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float imax = std::numeric_limits<float>::max();
      int min_index = 0;
      SGM::vector_type vec = accumulated_cost_.at<SGM::vector_type>(y, x);

      for (int d = 0; d < DISP_RANGE; d++) {
        if (vec[d] < imax) {
          imax = vec[d];
          min_index = d;
        }
      }
      disparity.at<uchar>(y, x) = static_cast<uchar>(DIS_FACTOR * min_index);
    }
  }
};

void SGMFlow::consistencyCheck(cv::Mat disparityLeft, cv::Mat disparityLeftBackward, cv::Mat disparity, bool interpl) {

  const int width     = parameters_.width;
  const int height    = parameters_.height;
  const int win_radius = parameters_.win_radius;
  
  std::vector<cv::Point2i> occFalse;
  cv::Mat disparityWoConsistency(height, width, CV_8UC1);

  //based on forward
  for (int y = win_radius; y < height - win_radius; ++y) {
    for (int x = win_radius; x < width - win_radius; ++x) {
      disparityWoConsistency.at<uchar>(y, x) = static_cast<uchar>(disparityLeftBackward.at<uchar>(y, x) *
                                                                  visualDisparity);
      unsigned short disparityLeftValue = static_cast<unsigned short>(disparityLeft.at<uchar>(y, x));
      float newx = x + img_rotation_.at<vector_type>(y, x)[0];
      float newy = y + img_rotation_.at<vector_type>(y, x)[1];
      float distx = newx - epipole_left_.at<float>(0);
      float disty = newy - epipole_left_.at<float>(1);

      float L = sqrt(distx * distx + disty * disty);
      float d = L * ((float) Vmax * (float) disparityLeftValue / DISP_RANGE) /
                (1.0 - ((float) Vmax * (float) disparityLeftValue / DISP_RANGE));
      int xx = round(newx + d * translation_left_.at<vector_type>(newy, newx)[0]);
      int yy = round(newy + d * translation_left_.at<vector_type>(newy, newx)[1]);

      if ((xx >= win_radius) && (yy >= win_radius) && xx < (width - win_radius) && yy < (height - win_radius)) {
        unsigned short disparityLeftBackwardValue = static_cast<unsigned short>(disparityLeftBackward.at<uchar>(yy,
                                                                                                                xx));
        disparity.at<uchar>(y, x) = static_cast<uchar>(disparityLeftValue * visualDisparity);
        if (abs(disparityLeftBackwardValue - disparityLeftValue) > disparityThreshold) {
          disparity.at<uchar>(y, x) = static_cast<uchar>(Dinvd);
          occFalse.push_back(cv::Point2i(x, y));
        }
      }
    }
  }
  
  const int len = occFalse.size();

  std::vector<unsigned short> newDisparity(len);

  for (int i = 0; i < len; i++) {
    std::vector<int> neiborInfo;
    bool occlusion;
    int x = occFalse[i].x;
    int y = occFalse[i].y;

    if (interpl) {
      {

        float xx = x + translation_left_last_.at<vector_type>(y, x)[0];
        float yy = y + translation_left_last_.at<vector_type>(y, x)[1];
        int dirx = 0;
        int dirxDisparityPx = 0;
        while ((dirx <= 10) && (xx <= width - win_radius) && (yy <= height - win_radius) && (xx >= win_radius) &&
               (yy >= win_radius)) {
          if (disparity.at<uchar>(round(yy), round(xx)) == static_cast<uchar>(Dinvd)) {
            xx = xx + translation_left_last_.at<vector_type>(y, x)[0];
            yy = yy + translation_left_last_.at<vector_type>(y, x)[1];
            continue;
          }
          dirxDisparityPx += static_cast<int>(disparity.at<uchar>(round(yy), round(xx)));
          xx = xx + translation_left_last_.at<vector_type>(y, x)[0];
          yy = yy + translation_left_last_.at<vector_type>(y, x)[1];
          dirx++;
        }
        if (dirx != 0) { neiborInfo.push_back(round(dirxDisparityPx / (float) dirx)); }
      }

      {
        float xx = x - translation_left_last_.at<vector_type>(y, x)[0];
        float yy = y - translation_left_last_.at<vector_type>(y, x)[1];
        int dirx = 0;
        int dirxDisparityNx = 0;
        while ((dirx <= 10) && (xx <= width - win_radius) && (yy <= height - win_radius) && (xx >= win_radius) &&
               (yy >= win_radius)) {
          if (disparity.at<uchar>(round(yy), round(xx)) == static_cast<uchar>(Dinvd)) {
            xx = xx - translation_left_last_.at<vector_type>(y, x)[0];
            yy = yy - translation_left_last_.at<vector_type>(y, x)[1];
            continue;
          }
          dirxDisparityNx += static_cast<int>(disparity.at<uchar>(round(yy), round(xx)));
          xx = xx - translation_left_last_.at<vector_type>(y, x)[0];
          yy = yy - translation_left_last_.at<vector_type>(y, x)[1];
          dirx++;
        }
        if (dirx != 0) { neiborInfo.push_back(round(dirxDisparityNx / (float) dirx)); }
      }

      if (neiborInfo.size() == 2) {
        occlusion = fabs((neiborInfo[0] - neiborInfo[1]) / fmin(neiborInfo[0], neiborInfo[1])) > 0.2 ? true : false;
      }
      else { occlusion = false; }

      {
        int yy = y + 1;
        int diry = 0;
        int dirxDisparityPy;
        while ((diry < 1) && (yy <= height - win_radius)) {
          if (disparity.at<uchar>(yy, x) == static_cast<uchar>(Dinvd)) {
            yy++;
            continue;
          }
          dirxDisparityPy = static_cast<int>(disparity.at<uchar>(yy, x));
          yy++;
          diry++;
        }
        if (diry != 0) { neiborInfo.push_back(round(dirxDisparityPy / (float) diry)); }
      }

      {
        int yy = y - 1;
        int diry = 0;
        int dirxDisparityNy;
        while ((diry < 1) && (yy >= win_radius)) {
          if (disparity.at<uchar>(yy, x) == static_cast<uchar>(Dinvd)) {
            yy--;
            continue;
          }
          dirxDisparityNy = static_cast<int>(disparity.at<uchar>(yy, x));
          yy--;
          diry++;
        }
        if (diry != 0) { neiborInfo.push_back(round(dirxDisparityNy / (float) diry)); }

      }

      {
        int dirxy = 0;
        int yy = y + 1;
        int xx = x - 1;
        int dirxDisparityNxPy;
        while ((dirxy < 1) && (yy <= height - win_radius) && (xx >= win_radius)) {
          if (disparity.at<uchar>(yy, xx) == static_cast<uchar>(Dinvd)) {
            yy++;
            xx--;
            continue;
          }
          dirxDisparityNxPy = static_cast<int>(disparity.at<uchar>(yy, xx));
          yy++;
          xx--;
          dirxy++;
        }
        if (dirxy != 0) { neiborInfo.push_back(round(dirxDisparityNxPy / (float) dirxy)); }
      }

      {
        int dirxy = 0;
        int yy = y + 1;
        int xx = x + 1;
        int dirxDisparityPxPy;
        while ((dirxy < 1) && (yy <= height - win_radius) && (xx <= width - win_radius)) {
          if (disparity.at<uchar>(yy, xx) == static_cast<uchar>(Dinvd)) {
            yy++;
            xx++;
            continue;
          }
          dirxDisparityPxPy = static_cast<int>(disparity.at<uchar>(yy, xx));
          yy++;
          xx++;
          dirxy++;
        }
        if (dirxy != 0) { neiborInfo.push_back(round(dirxDisparityPxPy / (float) dirxy)); }
      }

      {
        int dirxy = 0;
        int yy = y - 1;
        int xx = x + 1;
        int dirxDisparityPxNy;
        while ((dirxy < 1) && (yy >= win_radius) && (xx <= width - win_radius)) {
          if (disparity.at<uchar>(yy, xx) == static_cast<uchar>(Dinvd)) {
            yy--;
            xx++;
            continue;
          }
          dirxDisparityPxNy = static_cast<int>(disparity.at<uchar>(yy, xx));
          yy--;
          xx++;
          dirxy++;
        }
        if (dirxy != 0) { neiborInfo.push_back(round(dirxDisparityPxNy / (float) dirxy)); }
      }

      {
        int dirxy = 0;
        int yy = y - 1;
        int xx = x - 1;
        int dirxDisparityNxNy;
        while ((dirxy < 1) && (yy >= win_radius) && (xx >= win_radius)) {
          if (disparity.at<uchar>(yy, xx) == static_cast<uchar>(Dinvd)) {
            yy--;
            xx--;
            continue;
          }
          dirxDisparityNxNy = static_cast<int>(disparity.at<uchar>(yy, xx));
          yy--;
          xx--;
          dirxy++;
        }
        if (dirxy != 0) { neiborInfo.push_back(round(dirxDisparityNxNy / (float) dirxy)); }
      }
      
      std::sort(neiborInfo.begin(), neiborInfo.end());
      int secLow = neiborInfo[2];
      int median = neiborInfo[floor(neiborInfo.size() / 2.f)];
      unsigned short newValue = 0;
      if (occlusion == true) {
        newValue = secLow;
      } else {
        newValue = median;
      }
      newDisparity[i] = newValue;
    }

    for (int i = 0; i < len; i++) {
      int x = occFalse[i].x;
      int y = occFalse[i].y;
      disparity.at<uchar>(y, x) = static_cast<uchar>(newDisparity[i]);
    }
  }
};

void SGMFlow::computeCostRight() {

  const int width     = parameters_.width;
  const int height    = parameters_.height;
  const int win_radius = parameters_.win_radius;
  
  for (int y = win_radius; y < height - win_radius; ++y) {
    for (int x = win_radius; x < width - win_radius; ++x) {

      for (int w = 0; w < DISP_RANGE; w++) {

        for (int neiY = y - win_radius; neiY <= y + win_radius; neiY++) {
          for (int neiX = x - win_radius; neiX <= x + win_radius; neiX++) {
            float newx = neiX + img_rotation_backward_.at<vector_type>(neiY, neiX)[0];
            float newy = neiY + img_rotation_backward_.at<vector_type>(neiY, neiX)[1];
            float distx = newx - epipole_left_last_.at<float>(0);
            float disty = newy - epipole_left_last_.at<float>(1);
            //float distx = newx - epipole_left_.at<float>(0);
            //float disty = newy - epipole_left_.at<float>(1);

            float L = sqrt(distx * distx + disty * disty);
            //float d = L * ((float)Vmax*(float)w/DISP_RANGE);										
            float d = L * ((float) Vmax * (float) w / DISP_RANGE) / (1.0 + ((float) Vmax * (float) w / DISP_RANGE));

            int xx = round(newx - d * translation_left_last_.at<vector_type>(newy, newx)[0]);
            int yy = round(newy - d * translation_left_last_.at<vector_type>(newy, newx)[1]);

            if ((xx >= win_radius) && (yy >= win_radius) && xx < (width - win_radius) && yy < (height - win_radius)) {
              cost_right_to_left_.at<SGM::vector_type>(y, x)[w] +=
              fabs(derivative_flow_left_.at<float>(neiY, neiX) - derivative_flow_left_last_.at<float>(yy, xx))
              + (float) CENSUS_W * computeHammingDist(census_image_left_.at<uchar>(neiY, neiX), census_image_left_last_.at<uchar>(yy, xx));
            } else {
              cost_right_to_left_.at<SGM::vector_type>(y, x)[w] = 100000; //costRight.at<SGM::VecDf>(y,x)[w-1];
            }
          }
        }
      }
    }
  }
};

void SGMFlow::computeCost() {
  
  const int width     = parameters_.width;
  const int height    = parameters_.height;
  const int win_radius = parameters_.win_radius;
  
  for (int y = win_radius; y < height - win_radius; ++y) {
    for (int x = win_radius; x < width - win_radius; ++x) {
      for (int w = 0; w < DISP_RANGE; w++) {
        for (int neiY = y - win_radius; neiY <= y + win_radius; neiY++) {
          for (int neiX = x - win_radius; neiX <= x + win_radius; neiX++) {
            float newx = neiX + img_rotation_.at<vector_type>(neiY, neiX)[0];
            float newy = neiY + img_rotation_.at<vector_type>(neiY, neiX)[1];
            float distx = newx - epipole_left_.at<float>(0);
            float disty = newy - epipole_left_.at<float>(1);
            float L = sqrt(distx * distx + disty * disty);
            float d = L * ((float) Vmax * (float) w / DISP_RANGE) / (1.0 - ((float) Vmax * (float) w / DISP_RANGE));
            int xx = round(newx + d * translation_left_.at<vector_type>(newy, newx)[0]);
            int yy = round(newy + d * translation_left_.at<vector_type>(newy, newx)[1]);

            if ((xx >= win_radius) && (yy >= win_radius) && xx < (width - win_radius) && yy < (height - win_radius)) {
              cost_left_to_right_.at<SGM::vector_type>(y, x)[w] +=
              fabs(derivative_flow_left_last_.at<float>(neiY, neiX) - derivative_flow_left_.at<float>(yy, xx))
              + (float) CENSUS_W *
                computeHammingDist(census_image_left_last_.at<uchar>(neiY, neiX), census_image_left_.at<uchar>(yy, xx));
            } else {
              cost_left_to_right_.at<SGM::vector_type>(y, x)[w] = cost_left_to_right_.at<SGM::vector_type>(y, x)[w - 1];
            }
          }
        }
      }
    }
  }
};

void SGMFlow::computeFlow(cv::Mat &disparity, std::string fileName) {

  const int width     = parameters_.width;
  const int height    = parameters_.height;
  const int win_radius = parameters_.win_radius;
  
  float *out = (float *) calloc(height * width * 3, sizeof(float));

  for (int x = win_radius; x < width - win_radius; ++x) {
    for (int y = win_radius; y < height - win_radius; ++y) {
      if (disparity.at<uchar>(y, x) == static_cast<uchar>(Dinvd)) {
        out[(width * y * 3) + (x * 3) + 2] = static_cast<float>(0);

      } else {
        unsigned short disparityLeftValue = static_cast<unsigned short>(disparity.at<uchar>(y, x));
        float newx = x + img_rotation_.at<vector_type>(y, x)[0];
        float newy = y + img_rotation_.at<vector_type>(y, x)[1];
        float distx = newx - epipole_left_.at<float>(0);
        float disty = newy - epipole_left_.at<float>(1);

        float L = sqrt(distx * distx + disty * disty);
        float d = L * ((float) Vmax * (float) disparityLeftValue / DISP_RANGE) /
                  (1.0 - ((float) Vmax * (float) disparityLeftValue / DISP_RANGE));
        float Vx = d * translation_left_.at<vector_type>(newy, newx)[0] + img_rotation_.at<vector_type>(y, x)[0];
        float Vy = d * translation_left_.at<vector_type>(newy, newx)[1] + img_rotation_.at<vector_type>(y, x)[1];

        out[(width * y * 3) + (x * 3) + 0] = Vx;
        out[(width * y * 3) + (x * 3) + 1] = Vy;
        out[(width * y * 3) + (x * 3) + 2] = static_cast<float>(1);
      }
    }
  }

  std::string fileColorName = "color" + fileName;
  FlowImage fi(out, width, height);
  fi.write(fileName);
  free(out);
}

}



