#include "ukf.h"
#include "Eigen/Dense"
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

/**
 * BOOLEANS
 */
  
  // Initially, the system is not initialized
  is_initialized_ = false;
  
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  
  
/**
 * CONSTANTS
 */
 
  // Dimension of state vector
  n_x_ = 5;
  
  // Dimension of augmented state vector, including variables for the process noise (accelerations)
  n_aug_ = n_x_ + 2;
  
  // Spreading parameter used when generating sigma points
  lambda_ = 3 - n_x_;
  
  // Coefficient for A matrix
  A_coeff_ = sqrt(lambda_+n_aug_);
  

/**
 * VECTORS AND MATRICES
 */
 
  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  
  // Process noise covariance matrix
  Q_ = MatrixXd(2, 2);
  
  // Radar variances
  radar_variances_ = VectorXd(3);
  
  // Laser variances
  laser_variances_ = VectorXd(2);
  
  // Matrix of predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
  
  // Weights used to determine mean and covariance of generated sigma points
  weights_ = VectorXd(2*n_aug_+1);

  
/**
 * PARAMETERS
 */

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = .5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/8.0;
  
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
  

/**
 * NIS VARIABLES
 */
 
  // Number of radar measurements
  num_radar_meas = 0.0;
  
  // Number of radar above NIS threshold
  num_radar_NIS = 0;
  
  // Number of laser measurements
  num_laser_meas = 0.0;
  
  // Number of laser above NIS threshold
  num_laser_NIS = 0;

  
/**
 * CONSTANT VECTORS AND MATRICES
 */
 
  // Process noise covariance matrix
  Q_ << std_a_*std_a_, 0, 
        0, std_yawdd_*std_yawdd_;
		
  radar_variances_ << std_radr_*std_radr_, std_radphi_*std_radphi_, std_radrd_*std_radrd_;
  
  laser_variances_ << std_laspx_*std_laspx_, std_laspy_*std_laspy_;
  
  // Generate the weights
  float c2 = lambda_+n_aug_;
  weights_.fill(1/(2.*c2));
  weights_(0) = lambda_/(c2);
  
  
/**
 * GLOBAL TIMESTAMP
 */
  time_us_ = 0.0;
}


UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
  // If not yet initialized, initialize the state vector 
  // based on the measurement type
  if (!is_initialized_) {
	x_ << 0, 0, 0, 0, 0;
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
	  const float rho = meas_package.raw_measurements_[0];
	  const float phi = meas_package.raw_measurements_[1];
	  const float rho_dot = meas_package.raw_measurements_[2];
	  const float px = rho*cos(phi);
	  const float py = rho*sin(phi);
	  // Derive vx and by by taking derivatives of px and py. Since we assume d(phi)/dt = 0 (inertial), the second term in derivative goes to 0
	  const float v = abs(rho_dot);
	  x_ << px, py, v, 0, 0;
	  // Since radar measurements more accurately measure velocity, the expected error for v is relatively low
	  P_ << .75, 0, 0, 0, 0,
			0, .75, 0, 0, 0,
			0, 0, .2, 0, 0,
			0, 0, 0, 1.5, 0,
			0, 0, 0, 0, 1.;
	  cout << "Initialized with radar measurement" << endl;
	}
	
	else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
	  x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
	  // Since laser measurements more accurately measure position, the expected error for px and py are relatively low
	  P_ << .2, 0, 0, 0, 0,
			0, .2, 0, 0, 0,
			0, 0, .5, 0, 0,
			0, 0, 0, 1., 0,
			0, 0, 0, 0, 1.;
	  cout << "Initialized with laser measurement" << endl;
	}
	
	time_us_ = meas_package.timestamp_;
	is_initialized_ = true;
	return;
  }
  
  // Else, if already initialized, run the prediction step and then the update step
  double delta_t = (meas_package.timestamp_ - time_us_)/1000000.0;
  Prediction(delta_t);
  time_us_ = meas_package.timestamp_;
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
	UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
	UpdateRadar(meas_package);
  } 
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Prediction step:
  1. Generate the augmented state and covariance matrices. 
  2. Generate the augmented sigma points of the current state and covariance
  3. Generate the predicted sigma points, i.e. generate the sigma points from which we will pull a predicted state and covariance matrix
  4. Using the predicted sigma points, generate a predicted state and covariance matrix
  */
  
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
  
  // Initialize the augmented state vector, x_aug. Since mean of process noise is 0, set these terms equal to 0.
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_+1) = 0;

  // Generate the augmented covariance matrix (which includes the process noise), P_aug
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2,2) = Q_;
  
  // Generate the augmented sigma points of the current state
  MatrixXd A = P_aug.llt().matrixL();
  
  Xsig_aug.col(0) = x_aug;
  
  MatrixXd Acoeff = A*A_coeff_;
  for (int i = 0; i < n_aug_; i++) {
      Xsig_aug.col(i+1) = x_aug + Acoeff.col(i);
      Xsig_aug.col(n_aug_+1+i) = x_aug - Acoeff.col(i);
  }
  
  /**
   * Generate the predicted sigma points.
   * For each predicted sigma point, 
   * use it to update the prediction of the state vector
   */
  float c1 = 0.5*delta_t*delta_t;
  
  // x_pred is the predicted state vector
  VectorXd x_pred = VectorXd(n_x_);
  x_pred.fill(0.0);
  
  for (int i = 0; i < 2*n_aug_+1; i++) {
      VectorXd x_aug = Xsig_aug.col(i);
      VectorXd x = x_aug.head(5);
      VectorXd nu = x_aug.tail(2);
      const float px = x(0);
      const float py = x(1);
      const float v = x(2);
      const float psi = x(3);
      const float psid = x(4);
      const float nu_a = nu(0);
      const float nu_yaw = nu(1);

      VectorXd v1 = VectorXd(5);
      if (psid == 0) {
          v1 << v*cos(psi)*delta_t,
                v*sin(psi)*delta_t, 
                0, 
                0, 
                0;
      }
      else {
          v1 << (v/psid)*(sin(psi+psid*delta_t) - sin(psi)),
                (v/psid)*(-cos(psi+psid*delta_t) + cos(psi)),
                0,
                psid*delta_t,
                0;
      }
	  
	  VectorXd v2 = VectorXd(5);
      v2 << c1*cos(psi)*nu_a,
            c1*sin(psi)*nu_a,
            delta_t*nu_a,
            c1*nu_yaw,
            delta_t*nu_yaw;

      Xsig_pred_.col(i) = x + v1 + v2;
	  
	  // Update the predicted state vector
	  x_pred += weights_(i)*Xsig_pred_.col(i);
  }
  /**
   * Predict the covariance matrix
   */
  
  // P_pred is the predicted covariance matrix
  MatrixXd P_pred = MatrixXd(n_x_,n_x_);
  P_pred.fill(0.0);
  
  for (int i = 0; i < 2*n_aug_+1; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_pred += weights_(i)*x_diff*x_diff.transpose();
  }
  
  // Update current state vector and covariance matrix with the predicted ones
  x_ = x_pred;
  P_ = P_pred; 
}



 /**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
/*
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  
  const float px_meas = meas_package.raw_measurements_[0];
  const float py_meas = meas_package.raw_measurements_[1];
  
  const int n_z = 2;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  
  //actual measurement
  VectorXd z = VectorXd(n_z);
  z << px_meas, py_meas;
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  
  // Vector for singular predicted sigma point
  VectorXd xsig_pred = VectorXd(n_x_);
  
  for (int i = 0; i < 2*n_aug_+1; i++) {
    xsig_pred = Xsig_pred_.col(i);
    const float px = xsig_pred(0);
    const float py = xsig_pred(1);
   
    VectorXd z_sig = VectorXd(n_z);
    z_sig << px, py;
    Zsig.col(i) = z_sig;
    z_pred += weights_(i)*z_sig;
  }
  
  S += laser_variances_.asDiagonal();
  
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  
  for (int i = 0; i < 2*n_aug_+1; i++) {
      VectorXd z_diff = Zsig.col(i) - z_pred;
      S += weights_(i)*z_diff*z_diff.transpose();
	  
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      while (x_diff(1)> M_PI) x_diff(1)-=2.*M_PI;
      while (x_diff(1)<-M_PI) x_diff(1)+=2.*M_PI;
      Tc += weights_(i)*x_diff*z_diff.transpose();
  }
  
  MatrixXd K = Tc*S.inverse();
  
  x_ = x_ + K*(z-z_pred);
  P_ = P_ - K*S*K.transpose();
  
  const float nis = (z - z_pred).transpose()*S.inverse()*(z-z_pred);
  if (nis > 5.991) {
	num_laser_NIS += 1;
  }
  num_laser_meas += 1;
  cout << "Laser NIS percentage below threshold: %" << 100*(num_laser_meas-num_laser_NIS)/num_laser_meas << endl;
}
 */
 
/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  
  const float px_meas = meas_package.raw_measurements_[0];
  const float py_meas = meas_package.raw_measurements_[1];
  
  MatrixXd H = MatrixXd(2,5);
  H << 1, 0, 0, 0, 0,
	   0, 1, 0, 0, 0;
  MatrixXd R = MatrixXd(2,2);	   
  R << 0.0225, 0,
        0, 0.0225;
  VectorXd z = VectorXd(2);
  z << px_meas, py_meas;
  VectorXd z_pred = H*x_;
  VectorXd y = z - z_pred; 

  MatrixXd Ht = H.transpose();
  MatrixXd PHt = P_*Ht;
  MatrixXd S = H*PHt + R;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt*Si;

  // new estimate
  x_ = x_ + (K*y);
  const int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K*H) * P_;
  
  const float nis = (z - z_pred).transpose()*S.inverse()*(z-z_pred);
  if (nis > 5.991) {
	num_laser_NIS += 1;
  }
  num_laser_meas += 1;
  cout << "Laser NIS percentage below threshold: %" << 100*(num_laser_meas-num_laser_NIS)/num_laser_meas << endl;
}


/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  
  const float rho_meas = meas_package.raw_measurements_[0];
  const float phi_meas = meas_package.raw_measurements_[1];
  const float rhod_meas = meas_package.raw_measurements_[2];
  
  const int n_z = 3;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  
  //actual measurement
  VectorXd z = VectorXd(n_z);
  z << rho_meas, phi_meas, rhod_meas;
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  

  // Vector for singular predicted sigma point
  VectorXd xsig_pred = VectorXd(n_x_);

  //transform sigma points into measurement space
  //calculate mean predicted measurement
  //calculate innovation covariance matrix S
  for (int i = 0; i < 2*n_aug_+1; i++) {
    xsig_pred = Xsig_pred_.col(i);
    const float px = xsig_pred(0);
    const float py = xsig_pred(1);
    const float v = xsig_pred(2);
    const float psi = xsig_pred(3);
    const float psid = xsig_pred(4);
    
    const float rho_sig = sqrt(px*px + py*py);
    const float phi_sig = atan2(py,px);
    const float rhod_sig = (v*(px*cos(psi) + py*sin(psi)))/rho_sig; 
    VectorXd z_sig = VectorXd(n_z);
    z_sig << rho_sig, phi_sig, rhod_sig;
    Zsig.col(i) = z_sig;
    z_pred += weights_(i) * z_sig;
  }
  
  /** Generate the S and Tc (cross correlation) matrices
    * calculate Kalman gain K;
    * update state mean and covariance matrix
    */
  S += radar_variances_.asDiagonal();
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {
      VectorXd z_diff = Zsig.col(i) - z_pred;
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
      S += weights_(i)*z_diff*z_diff.transpose();
	  
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      while (x_diff(1)> M_PI) x_diff(1)-=2.*M_PI;
      while (x_diff(1)<-M_PI) x_diff(1)+=2.*M_PI;
      Tc += weights_(i)*x_diff*z_diff.transpose();
  }
  
  MatrixXd K = Tc*S.inverse();
  
  x_ = x_ + K*(z-z_pred);
  P_ = P_ - K*S*K.transpose();
  
  const float nis = (z - z_pred).transpose()*S.inverse()*(z-z_pred);
  if (nis > 7.815) {
	num_radar_NIS += 1;
  }
  num_radar_meas += 1;
  cout << "Radar NIS percentage below threshold: %" << 100*(num_radar_meas-num_radar_NIS)/num_radar_meas << endl;
}
