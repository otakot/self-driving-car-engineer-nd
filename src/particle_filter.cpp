/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

constexpr int TOTAL_PARTICLES = 100;
constexpr double DEFAULT_WEIGHT = 1.0;
constexpr double EPSILON = 0.001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.

  num_particles_ = TOTAL_PARTICLES;

  default_random_engine gen;

  // emulate noisy GPS position
  normal_distribution<double> particle_x(x, std[0]);
  normal_distribution<double> particle_y(y, std[1]);
  normal_distribution<double> particle_theta(theta, std[2]);

  for (int i = 0; i < TOTAL_PARTICLES; ++i) {

    // create particle from "noisy" GPS data with initial weight
    Particle particle = {i, particle_x(gen), particle_y(gen), particle_theta(gen), DEFAULT_WEIGHT};

    weights_.push_back(DEFAULT_WEIGHT);
    particles_.push_back(particle);
  }

  this->is_initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.

  default_random_engine gen;

  for (int i = 0; i < num_particles_; ++i) {
    double particle_x = particles_[i].x;
    double particle_y = particles_[i].y;
    double particle_theta = particles_[i].theta;

    double predicted_x;
    double predicted_y;
    double predicted_theta;

    const double delta_theta = yaw_rate * delta_t;

    // check for straight line movement
    if (fabs(yaw_rate) < EPSILON) {
      predicted_x = particle_x + velocity * delta_t * cos(particle_theta);
      predicted_y = particle_y + velocity * delta_t * sin(particle_theta);
      predicted_theta = particle_theta;
    } else {
      predicted_x = particle_x + (velocity / yaw_rate) * (sin(particle_theta + delta_theta) - sin(particle_theta));
      predicted_y = particle_y + (velocity / yaw_rate) * (cos(particle_theta) - cos(particle_theta + delta_theta));
      predicted_theta = particle_theta + delta_theta;
    }

    normal_distribution<double> norm_dist_x(predicted_x, std_pos[0]);
    normal_distribution<double> norm_dist_y(predicted_y, std_pos[1]);
    normal_distribution<double> norm_dist_theta(predicted_theta, std_pos[2]);

    particles_[i].x = norm_dist_x(gen);
    particles_[i].y = norm_dist_y(gen);
    particles_[i].theta = norm_dist_theta(gen);
  }


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted_landmarks,
                                     std::vector<LandmarkObs>& observations) {
  // Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.

  for(LandmarkObs& observation : observations){

    // set unrealistic initial values
    double min_distance = numeric_limits<double>::max();
    int closest_landmark_id = -1;

    for(LandmarkObs& landmark : predicted_landmarks){

      const double distance = dist(landmark.x, landmark.y, observation.x, observation.y);

      if (distance < min_distance) {
        closest_landmark_id = landmark.id;
        min_distance = distance;
      }
    }
    observation.id = closest_landmark_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	// Update the weights of each particle using a mult-variate Gaussian distribution.
  for(Particle& particle : particles_){

      // 1. select only landmarks within car's sensor range from the particle
      vector<LandmarkObs> predicted_landmarks;
      for(const Map::single_landmark_s& landmark : map_landmarks.landmark_list){

        // calculate distance from particle to landmark
        double distance_to_landmark = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);

        if(distance_to_landmark < sensor_range) {
          predicted_landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
        }
      }

      // 2. translate observations coordinates from vehicle to map coordinate system
      vector<LandmarkObs> translated_observations;

      const double sin_theta = sin(particle.theta);
      const double cos_theta = cos(particle.theta);

      for(const LandmarkObs& observation : observations){
        LandmarkObs translated_observation;
        translated_observation.id = observation.id;
        translated_observation.x = particle.x + observation.x * cos_theta - observation.y * sin_theta;
        translated_observation.y = particle.y + observation.x * sin_theta + observation.y * cos_theta;
        translated_observations.push_back(translated_observation);
      }

      // 3. identify the closest landmark for every observation
      dataAssociation(predicted_landmarks, translated_observations);

      // 4. calculate the weight of particle using Multivariate Gaussian distribution.*/
      particle.weight = 1.0; // initial value of weight

      const double sigma_x = std_landmark[0];
      const double sigma_y = std_landmark[1];
      const double gaussian_normalizer = 1/(2 * M_PI * sigma_x * sigma_y);

      const double double_sigma_x_2 = pow(sigma_x, 2) * 2;
      const double double_sigma_y_2 = pow(sigma_y, 2) * 2;


      for(const LandmarkObs& observation : translated_observations){

        // since landmark ids in a map_data.txt file are sorted and start from 1
        // find the landmark on a map with same id as observation has
        Map::single_landmark_s landmark = map_landmarks.landmark_list[observation.id]; // id -1 ?

        // multivariate Gaussian distribution method
        double exp_x = pow((observation.x - landmark.x_f), 2) / double_sigma_x_2;
        double exp_y = pow((observation.y - landmark.y_f), 2) / double_sigma_y_2;
        double weight = gaussian_normalizer * exp(-(exp_x + exp_y));

        particle.weight *=  weight;
      }

      // TODO: normalize weights?

      weights_.push_back(particle.weight);
  }
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their weight.

  vector<Particle> resampled_particles;

  std::default_random_engine gen;
  std::discrete_distribution<int> weighted_distribution(weights_.begin(), weights_.end());

  for (int i = 0; i < num_particles_; ++i) {

    const int particle_id = weighted_distribution(gen);
    resampled_particles.push_back(particles_[particle_id]);
  }

  particles_ = resampled_particles;

  // reset weights
  weights_.clear();
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
