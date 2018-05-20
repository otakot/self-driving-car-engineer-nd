#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  //  initializing matrices

  //measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
                0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0,      0,
                0,    0.0009, 0,
                0,    0,      0.09;


  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;


  // state transition matrix
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

  //covariance matrix.
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0,    0,
             0, 1, 0,    0,
             0, 0, 1000, 0,
             0, 0, 0,    1000;

  // acceleration noise
  noise_ax = 9.0;
  noise_ay = 9.0;

}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       Convert radar from polar to cartesian coordinates and initialize state.
       */

      double rho = measurement_pack.raw_measurements_[0];  // range
      double phi = measurement_pack.raw_measurements_[1];  // bearing
      double rho_dot = measurement_pack.raw_measurements_[2];  // velocity of rho

      // Coordinates convertion from polar to cartesian
      double x = rho * cos(phi);
      double y = rho * sin(phi);

      double vx = rho_dot * cos(phi);
      double vy = rho_dot * sin(phi);
      ekf_.x_ << x, y, vx, vy;
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
       Initialize state.
       */
      double x = measurement_pack.raw_measurements_[0];
      double y = measurement_pack.raw_measurements_[1];
      ekf_.x_ << x, y, 0, 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  // State transition matrix update
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Noise covariance matrix computation

  double dt_2 = dt * dt;      // dt^2
  double dt_3 = dt_2 * dt;    // dt^3
  double dt_4 = dt_3 * dt;    // dt^4
  double dt_4_div_4 = dt_4 / 4;   // dt^4/4
  double dt_3_div_2 = dt_3 / 2;   // dt^3/2

  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4_div_4 * noise_ax, 0,                     dt_3_div_2 * noise_ax, 0,
             0,                     dt_4_div_4 * noise_ay, 0,                     dt_3_div_2 * noise_ay,
             dt_3_div_2 * noise_ax, 0,                     dt_2 * noise_ax,       0,
             0,                     dt_3_div_2 * noise_ay, 0,                     dt_2 * noise_ay;

  ekf_.Predict();

  cout << "Predicted x_ = " << ekf_.x_ << endl;
  cout << "Predicted P_ = " << ekf_.P_ << endl;

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "Updated x_ = " << ekf_.x_ << endl;
  cout << "Updated P_ = " << ekf_.P_ << endl;
}
