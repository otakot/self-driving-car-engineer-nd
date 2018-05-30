#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

namespace tools {

  /**
  * Calculates root mean squared error.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
   * Calculates NormalizedInnovationSquared
   *
   * @param z_diff difference between predicted and actual measurement
   * @param s measurement covariance matrix
   */
  VectorXd CalculateNIS(const MatrixXd& z_diff, const MatrixXd& s);

} // namespace tools

#endif /* TOOLS_H_ */
