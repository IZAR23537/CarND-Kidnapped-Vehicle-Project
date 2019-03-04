#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;


void ParticleFilter::init(double x, double y, double theta, double std[]) {

	// Number of particles
	num_particles = 100;  
	
	// Generator
	std::default_random_engine gen; 
	
	// Normal distribution for x, y and theta
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);
	
	
	// Initialize particles
	for (int i = 0; i < num_particles; i++){
		
		Particle p;
		
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
									
	
	// Generator
	std::default_random_engine gen;

	double new_theta;
	double new_x_mean;
	double new_y_mean;
    
	// Add measurements to each particle and add random Gaussian noise
	for (int i = 0; i < num_particles; i++) {
		
		if (fabs(yaw_rate) < 0.0001) {
			
			new_theta  = particles[i].theta;
			new_x_mean = particles[i].x + (velocity * delta_t * cos(new_theta));
			new_y_mean = particles[i].y + (velocity * delta_t * sin(new_theta));
			
		} else {
			
			new_theta  = particles[i].theta + yaw_rate * delta_t;
			new_x_mean = particles[i].x + (velocity / yaw_rate) * (sin(new_theta) - sin(particles[i].theta));
			new_y_mean = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(new_theta));
		}

		// Normal distribution for x, y and theta 
		std::normal_distribution<double> dist_x(new_x_mean, std_pos[0]);
		std::normal_distribution<double> dist_y(new_y_mean, std_pos[1]);
		std::normal_distribution<double> dist_theta(new_theta, std_pos[2]);

		particles[i].x =  dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
   
    for (unsigned int i = 0; i < observations.size(); i++) {
    
		// Set the minimum distance value
		double min_distance = 1000000000;
    
		int map_id = 0;

		for (unsigned int j = 0; j < predicted.size(); j++) {
		
			// Calculate the distance between all of the observations and predictions 
			double distance = sqrt(pow(observations[i].x - predicted[j].x, 2.0) + pow(observations[i].y - predicted[j].y, 2.0));
		
			// Select Nearest Neighbor 
			if (distance < min_distance) {
			
				min_distance = distance;
				map_id = predicted[j].id;
			}
		}	
		
		observations[i].id = map_id;
	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
   
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	
	for (int i = 0; i < num_particles; i++) {
		
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
	
		vector<LandmarkObs> landmark_pred;
		
		// Check landmarks which is in range
		for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
		
			double x_p = map_landmarks.landmark_list[j].x_f;
			double y_p = map_landmarks.landmark_list[j].y_f;
			int id_p = map_landmarks.landmark_list[j].id_i;

			if (pow(x - x_p, 2) + pow(y - y_p, 2) <= pow(sensor_range,2)) {
				
				landmark_pred.push_back(LandmarkObs{id_p, x_p, y_p});
			}
		}

		// Transform observations coordinates to map's coordinate.
		vector<LandmarkObs> transformed;
		
		for (unsigned int k = 0; k < observations.size(); k++) {
			
			double transformed_x = cos(theta) * observations[k].x - sin(theta) * observations[k].y + x;
			double transformed_y = sin(theta) * observations[k].x + cos(theta) * observations[k].y + y;
			transformed.push_back(LandmarkObs{observations[k].id, transformed_x, transformed_y});
		}
		
		// Associate landmarks with observations
		dataAssociation(landmark_pred, transformed);
   
		particles[i].weight = 1.0;
		
		for (unsigned int l = 0; l < transformed.size(); l++){
			
			double observation_x = transformed[l].x;
			double observation_y = transformed[l].y;
			int observation_id = transformed[l].id;
			
			double landmark_x, landmark_y;
			
			for (unsigned int m = 0; m < landmark_pred.size(); m++){
				
				if(observation_id == landmark_pred[m].id){
					
					landmark_x = landmark_pred[m].x;
					landmark_y = landmark_pred[m].y;
				}
			}
			
			// Calculate new weight
			double distance_X = landmark_x - observation_x;
			double distance_Y = landmark_y - observation_y;
			double weight = (1 / (2 * M_PI * std_x * std_y)) * exp(-(pow(distance_X, 2) / (2 * pow(std_x, 2)) + (pow(distance_Y, 2) / (2 * pow(std_y, 2)))));
			
			particles[i].weight *= weight;
		}
	}
} 
   
void ParticleFilter::resample() { 

	/* This  implementation is based on huboqiang's resample implementation:  https://github.com/huboqiang/CarND-Kidnapped-Vehicle-Project/blob/master/src/particle_filter.cpp */
	
	std::default_random_engine gen;  
	vector<Particle> new_particle;
	vector<double> weights;
  
	for (int i = 0; i < num_particles; i++) {
		
		weights.push_back(particles[i].weight);
	}
  
	// Implement resampling wheel 
	
	std::discrete_distribution<int> init_index(weights.begin(), weights.end());
	
	double max_weight = *max_element(weights.begin(), weights.end());
	
	std::uniform_real_distribution<double> weight_dist(0.0, max_weight);
	
	int index = init_index(gen);
	
	double beta = 0.0;

	for (int i = 0; i < num_particles; i++) {
    
		beta += weight_dist(gen) * 2.0;
	
		while (weights[index] < beta ) {
		
			beta -= weights[index];
	  
			index = (index + 1) % num_particles;
		}
	
		new_particle.push_back(particles[index]);
	}
	
	particles = new_particle;
}


void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
										 
  // particle: the particle to which assign each listed association, 
  //   		   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  
	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
	
	vector<int> v = best.associations;
	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
  
	return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
	
	vector<double> v;

	if (coord == "X") {
		
		v = best.sense_x;
		
	} else {
	  
		v = best.sense_y;
	}

	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
  
	return s;
}