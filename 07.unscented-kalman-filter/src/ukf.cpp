#include "ukf.h"
#include "Eigen/Dense"
#include "tools.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // instantiate state vector
  x_ = VectorXd(5);

  // instantiate covariance matrix
  P_ = MatrixXd(5, 5);


  // Process noise standard deviation longitudinal acceleration in m/s^2
  // assuming maximum acceleration of cyclist will in 95% of time is
  // smaller than acceerating from 0km/h to 35km/h in 3 seconds
  std_a_ = 1.62;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // empirically selected
  std_yawdd_ = 0.6;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  //State vector dimensions
  n_x_ = 5;

  // define spreading factor
  lambda_ = 3 - n_x_;

  // Augmented state dimension
  n_aug_ = n_x_ + 2;

  // Number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // instantiate weights
  weights_ = VectorXd(n_sig_);

  // instantiate radar measurement noise covariance matrix
  R_radar_ = MatrixXd(3, 3);

  // instantiate lidar measurement noise covariance matrix
  R_lidar_ = MatrixXd(2, 2);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {

      // first measurement
      cout << "Initialization of UKF... " << endl;
      x_ << 1, 1, 1, 1, 1;

      if (meas_package.sensor_type_ == meas_package.RADAR) {
        /**
         Convert radar from polar to cartesian coordinates and initialize state.
         */

        double rho = meas_package.raw_measurements_[0];  // range
        double phi = meas_package.raw_measurements_[1];  // bearing
        double rho_dot = meas_package.raw_measurements_[2];  // velocity of rho

        // Coordinates convertion from polar to cartesian
        double px = rho * cos(phi);
        double py = rho * sin(phi);

        double vx = rho_dot * cos(phi);
        double vy = rho_dot * sin(phi);
        double v = sqrt(vx * vx + vy * vy);
        x_ << px, py, v, 0, 0;
      } else if (meas_package.sensor_type_ == meas_package.LASER) {
        /**
         Just Initialize state.
         */
        double px = meas_package.raw_measurements_[0];
        double py = meas_package.raw_measurements_[1];
        x_ << px, py, 0, 0, 0;
      }

      // Initialize weights
      weights_.fill(0.5 / (n_aug_ + lambda_));
      weights_(0) = lambda_ / (lambda_ + n_aug_);

      // Initialize covariance matrix
      P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;

      // Initialize measurement noise covariance matrices for radar and lidar
      R_radar_ << std_radr_ * std_radr_,   0,                         0,
                    0,                     std_radphi_ * std_radphi_, 0,
                    0,                     0,                         std_radrd_ * std_radrd_;

      R_lidar_ << std_laspx_ * std_laspx_, 0,
                  0,                       std_laspy_ * std_laspy_;

      // Store the initial timestamp for dt calculation
      time_us_ = meas_package.timestamp_;

      // done initializing, no need to predict or update
      is_initialized_ = true;
    } else {

      // Calculate the time between measurements in seconds
      double dt = (meas_package.timestamp_ - time_us_);
      dt /= 1000000.0; // convert microseconds to seconds
      time_us_ = meas_package.timestamp_;

      // Prediction step
      Prediction(dt);

      // Update step
      if (meas_package.sensor_type_ == meas_package.RADAR && use_radar_) {
         UpdateRadar(meas_package);
      }

      if (meas_package.sensor_type_ == meas_package.LASER && use_laser_) {
         UpdateLidar(meas_package);
      }

      // debugging output
      cout << "x_ = " << x_ << endl;
      cout << "P_ = " << P_ << endl;
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    // GENERATE SIGMA POINTS

    //create augmented state vector
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug.fill(0);
    x_aug.head(n_x_) = x_;

    //create augmented covariance matrix
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_,n_x_) = P_;
    P_aug(5,5) = std_a_*std_a_;
    P_aug(6,6) = std_yawdd_*std_yawdd_;

    // create sigma point matrix
    MatrixXd Xsig_aug = GenerateSigmaPoints(x_aug, P_aug, lambda_, n_sig_);


    // PREDICT SIGMA POINTS
    Xsig_pred_ = PredictSigmaPoints(Xsig_aug, delta_t, n_x_, n_sig_);


    // PREDICT STATE

    // calculate predicted state mean
    x_ = Xsig_pred_ * weights_;

    // calculate predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < n_sig_; i++) {  //iterate over sigma points

      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      //angle normalization
      while (x_diff(3)> M_PI)
        x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI)
        x_diff(3)+=2.*M_PI;

      P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  // measurement dimension, lidar can measure only px, py
  int n_z = 2;

  // lidar measurement vector
  VectorXd z = VectorXd(n_z);

  // read measurements
  double px = meas_package.raw_measurements_[0];
  double py = meas_package.raw_measurements_[1];

  z << px, py;

  // matrix for sigma points in measurement space
  MatrixXd z_sig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //transform predicted sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_sig(0, i) = Xsig_pred_(0, i);
    z_sig(1, i) = Xsig_pred_(1, i);
  }

  Update(z_sig, z, R_lidar_, meas_package.sensor_type_);

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // measurement dimension, radar can measure rho, phi and rho_dot
  int n_z = 3;

  // radar measurement vector
  VectorXd z = VectorXd(n_z);

  // read measurements
  double rho = meas_package.raw_measurements_[0];
  double phi = meas_package.raw_measurements_[1];
  double v = meas_package.raw_measurements_[2];

  z << rho, phi, v;

  // matrix for sigma points in measurement space
  MatrixXd z_sig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //transform predicted sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double vel = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double v_x = cos(yaw) * vel;
    double v_y = sin(yaw) * vel;

    // rho
    z_sig(0, i) = sqrt(p_x * p_x + p_y * p_y);

    // phi
    z_sig(1, i) = atan2(p_y, p_x);

    // rho_dot
    z_sig(2, i) = z_sig(0, i) < 0.0001 ?
        (p_x * v_x + p_y * v_y) / 0.0001 :
        (p_x * v_x + p_y * v_y) / z_sig(0, i);
  }

  Update(z_sig, z, R_radar_, meas_package.sensor_type_);

}

void UKF::Update(const MatrixXd& z_sig, const VectorXd& z, const MatrixXd& R, MeasurementPackage::SensorType sensor_type) {

  // predicted measurement vector
  VectorXd z_pred = VectorXd(z.size());

  // predicted measurement mean
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * z_sig.col(i);
  }

  // measurement covariance matrix
  MatrixXd S = MatrixXd(z.size(), z.size());
  S.fill(0.0);

  // Cross correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, z.size());
  Tc.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    // measurement difference
    VectorXd z_diff = z_sig.col(i) - z_pred;

    // angle normalization
    if (sensor_type == MeasurementPackage::RADAR) {
      while (z_diff(1)> M_PI)
        z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI)
        z_diff(1)+=2.*M_PI;
    }

    // update matrix
    S = S + weights_(i) * z_diff * z_diff.transpose();

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3)> M_PI)
      x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI)
      x_diff(3)+=2.*M_PI;

    // update matrix
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S = S + R;

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

  // normalize angle
  if (sensor_type == MeasurementPackage::RADAR) {
    while (z_diff(1)> M_PI)
          z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI)
         z_diff(1)+=2.*M_PI;
  }

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Calculate NIS
  string sensor_type_label =
      sensor_type == MeasurementPackage::RADAR ? "RADAR" : "LASER";
  VectorXd nis = tools::CalculateNIS(z_diff, S);
  cout << sensor_type_label << " NIS: " << nis << endl;
}

MatrixXd UKF::GenerateSigmaPoints(const VectorXd& x, const MatrixXd& P, double lambda, int n_sig) {

  //create sigma point matrix
  MatrixXd Xsig(x.size(), n_sig );

  //calculate square root of P
  MatrixXd A = P.llt().matrixL();

  Xsig.col(0) = x;

  double tmp = sqrt(lambda + x.size());
  for (int i = 0; i < x.size(); i++){
      Xsig.col( i + 1 ) = x + tmp * A.col(i);
      Xsig.col( i + 1 + x.size() ) = x - tmp * A.col(i);
  }
  return Xsig;
}

MatrixXd UKF::PredictSigmaPoints(const MatrixXd& Xsig, double delta_t, int n_x, int n_sig) {
  MatrixXd Xsig_pred(n_x, n_sig);
  //predict sigma points
  for (int i = 0; i< n_sig; i++)
  {
    //extract values for better readability
    double p_x = Xsig(0,i);
    double p_y = Xsig(1,i);
    double v = Xsig(2,i);
    double yaw = Xsig(3,i);
    double yawd = Xsig(4,i);
    double nu_a = Xsig(5,i);
    double nu_yawdd = Xsig(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

  return Xsig_pred;
}
