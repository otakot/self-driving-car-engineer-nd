#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  UpdateWithPositionError(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Define local variables to work on
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];

  // Define Radar variables
  float rho = sqrt(pow(px, 2) + pow(py, 2));
  float phi = atan2(py, px);  // The use of atan2 return values between -pi and pi instead of -pi/2 and pi/2 for atan
  float rhodot = (fabs(rho) < 0.0001) ? 0 : (vx * px + vy * py) / rho;

  VectorXd z_pred(3);
  z_pred << rho, phi, rhodot;

  // calculate position error
  VectorXd y = z - z_pred;

  // normalizie angles in a way that y[1] stays between -Pi and +Pi
  while (y[1] > M_PI) {
    y[1] -= (2 * M_PI);
  }
  while (y[1] < - M_PI) {
    y[1] += (2 * M_PI);
  }

  UpdateWithPositionError(y);
}

void KalmanFilter::UpdateWithPositionError(const VectorXd &y){
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;

  // calculate new state
  x_ = x_ + (K * y);

  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);

  // calculate new covariance
  P_ = (I - K * H_) * P_;
}
