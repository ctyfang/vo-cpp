#ifndef EXTRACTOR_H_
#define EXTRACTOR_H_

#include <vector>
#include <set>
#include <algorithm>
#include <iostream>

#include <boost/range/algorithm/set_algorithm.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/irange.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>
#include <eigen3/unsupported/Eigen/NumericalDiff>
#include <Eigen/Dense>

// Generic functor
// See http://eigen.tuxfamily.org/index.php?title=Functors
// C++ version of a function pointer that stores meta-data about the function
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{

  // Information that tells the caller the numeric type (eg. double) and size (input / output dim)
  typedef _Scalar Scalar;
  enum { // Required by numerical differentiation module
      InputsAtCompileTime = NX,
      ValuesAtCompileTime = NY
  };

  // Tell the caller the matrix sizes associated with the input, output, and jacobian
  typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
  typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
  typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

  // Local copy of the number of inputs
  int m_inputs, m_values;
  cv::Mat P1, P2;
  std::vector<cv::Point2f> kp_1, kp_2;

  // Two constructors:
  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  // Get methods for users to determine function input and output dimensions
  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

};

/***********************************************************************************************/

// https://en.wikipedia.org/wiki/Test_functions_for_optimization
// Booth Function
// Implement f(x,y) = (x + 2*y -7)^2 + (2*x + y - 5)^2
struct LMFunctor2 : Functor<double>
{
  // Simple constructor
  LMFunctor2(int m, int n): Functor<double>(m, n) {
     m_ = m;
     n_ = n;
  }

  // Implementation of the objective function
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const {
    /*
     * Evaluate the Booth function.
     * Important: LevenbergMarquardt is designed to work with objective functions that are a sum
     * of squared terms. The algorithm takes this into account: do not do it yourself.
     * In other words: objFun = sum(fvec(i)^2)
     */
      for (int i = 0; i < m_; i++) {
         float l_x = x(3*i);
         float l_y = x(3*i+1);
         float l_z = x(3*i+2);
         cv::Mat l_vec = cv::Mat::ones(4, 1, CV_32F);
         l_vec.at<float>(0, 0) = l_x;
         l_vec.at<float>(0, 1) = l_y;
         l_vec.at<float>(0, 2) = l_z;

         cv::Mat p1_vec = P1 * l_vec;
         p1_vec /= p1_vec.at<float>(0, 2);
         cv::Point2f kp1_proj(p1_vec.at<float>(0, 1), p1_vec.at<float>(0, 2));

         cv::Mat p2_vec = P2 * l_vec;
         p2_vec /= p2_vec.at<float>(0, 2);
         cv::Point2f kp2_proj(p2_vec.at<float>(0, 1), p2_vec.at<float>(0, 2));

         cv::Point2f diff_1 = kp1_proj - kp_1[i];
         cv::Point2f diff_2 = kp2_proj - kp_2[i];
         fvec(i) = cv::sqrt(diff_1.dot(diff_1)) + cv::sqrt(diff_2.dot(diff_2));
      }
    return 0;
  }

  cv::Mat P1 = cv::Mat::zeros(3, 4, CV_32F);
  cv::Mat P2 = cv::Mat::zeros(3, 4, CV_32F);
  std::vector<cv::Point2f> kp_1, kp_2;
  int m_;
  int n_;
};


struct LMFunctor
{
   // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
   int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const {
      // 'x' has dimensions n x 1
      // It contains the current estimates for the parameters.

      // 'fvec' has dimensions m x 1
      // It will contain the error for each data point. (Sum of squared reprojection errors for each landmark)

      for (int i = 0; i < values(); i++) {
         float l_x = x(3*i);
         float l_y = x(3*i+1);
         float l_z = x(3*i+2);
         cv::Mat l_vec = cv::Mat::ones(4, 1, CV_32F);
         l_vec.at<float>(0, 0) = l_x;
         l_vec.at<float>(0, 1) = l_y;
         l_vec.at<float>(0, 2) = l_z;

         cv::Mat p1_vec = P1 * l_vec;
         p1_vec /= p1_vec.at<float>(0, 2);
         cv::Point2f kp1_proj(p1_vec.at<float>(0, 1), p1_vec.at<float>(0, 2));

         cv::Mat p2_vec = P2 * l_vec;
         p2_vec /= p2_vec.at<float>(0, 2);
         cv::Point2f kp2_proj(p2_vec.at<float>(0, 1), p2_vec.at<float>(0, 2));

         cv::Point2f diff_1 = kp1_proj - kp_1[i];
         cv::Point2f diff_2 = kp2_proj - kp_2[i];
         fvec(i) = diff_1.dot(diff_1) + diff_2.dot(diff_2);
      }

      return 0;
   }

   // Compute the jacobian of the functions
   int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const {
      // 'x' has dimensions n x 1
      // It contains the current estimates for the parameters.

      // 'fjac' has dimensions m x n
      // It will contain the jacobian of the errors, calculated numerically in this case.

      float epsilon;
      epsilon = 1e-5f;

      for (int i = 0; i < x.size(); i++) {
         Eigen::VectorXf xPlus(x);
         xPlus(i) += epsilon;
         Eigen::VectorXf xMinus(x);
         xMinus(i) -= epsilon;

         Eigen::VectorXf fvecPlus(values());
         operator()(xPlus, fvecPlus);

         Eigen::VectorXf fvecMinus(values());
         operator()(xMinus, fvecMinus);

         Eigen::VectorXf fvecDiff(values());
         fvecDiff = (fvecPlus - fvecMinus) / (2.0f * epsilon);

         fjac.block(0, i, values(), 1) = fvecDiff;
      }

      return 0;
   }

   // Number of data points, i.e. values.
   int m;
   cv::Mat P1;
   cv::Mat P2;
   std::vector<cv::Point2f> kp_1;
   std::vector<cv::Point2f> kp_2;

   // Returns 'm', the number of values.
   int values() const { return m; }

   // The number of parameters, i.e. inputs.
   int n;

   // Returns 'n', the number of inputs.
   int inputs() const { return n; }
};

class Extractor
{
public:
   Extractor(const cv::Mat &K);
   void ExtractSIFT(std::shared_ptr<cv::Mat> frame, std::vector<cv::KeyPoint> &keypoints,
                    cv::Mat &descriptors);
   void ExtractCorners(std::shared_ptr<cv::Mat> frame, std::vector<cv::KeyPoint> &keypoints,
                       cv::Mat &descriptors);
   void MatchSIFT(const cv::Mat &descriptors_1, const cv::Mat &descriptors_2,
                  std::vector<std::vector<cv::DMatch>> &matches, const float ratio_threshold = 0.8);
   void EstimateMotion(std::vector<cv::Point2f> &keypoints_1,
                       std::vector<cv::Point2f> &keypoints_2,
                       cv::Mat &R, cv::Mat &t);
   void EstimateMotion(const std::vector<cv::Point2f> &keypoints,
                       const std::vector<cv::Point3f> &landmarks,
                       cv::Mat &R, cv::Mat &t);
   void FilterKeypoints(std::vector<cv::KeyPoint> &keypoints_1,
                        std::vector<cv::KeyPoint> &keypoints_2,
                        std::vector<std::vector<cv::DMatch>> &matches,
                        std::vector<cv::Point2f> &keypoints_1_m,
                        std::vector<cv::Point2f> &keypoints_2_m,
                        std::vector<cv::Point2f> &keypoints_1_nm,
                        std::vector<cv::Point2f> &keypoints_2_nm);
   cv::Point3f ComputePosition(cv::Mat &R, cv::Mat &t);
   std::vector<cv::Point3f> Triangulate(std::vector<cv::Point2f> keypoints_1,
                                        std::vector<cv::Point2f> keypoints_2,
                                        cv::Mat &R1, cv::Mat &t1,
                                        cv::Mat &R2, cv::Mat &t2);
   std::vector<uchar> TrackKeypoints(std::shared_ptr<cv::Mat> frame_curr,
                                     std::vector<cv::Point2f> &keypoints);
   void FilterKeypointsAndLandmarks(std::vector<cv::Point2f> &keypoints,
                                    std::vector<cv::Point3f> &landmarks,
                                    std::vector<uchar> &mask);

private:
   cv::Mat K_;
   cv::Mat frame_prev_;
   cv::Ptr<cv::SIFT> sift_;
   cv::Ptr<cv::DescriptorMatcher> matcher_;
};

#endif // EXTRACTOR_H_
