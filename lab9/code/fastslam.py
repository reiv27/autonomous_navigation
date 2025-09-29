from read_data import read_world, read_sensor_data
from misc_tools import *
import numpy as np
import math
import copy

# plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def initialize_particles(num_particles, num_landmarks):
    # initialize particle at pose [0, 0, 0] with an empty map

    particles = []

    for _ in range(num_particles):
        particle = dict()

        # initialize pose: at the beginning, robot is certain it is at [0, 0, 0]
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        # initial weight
        particle['weight'] = 1.0 / num_particles

        # particle history aka all visited poses
        particle['history'] = []

        # initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            # initialize the landmark mean and covariance
            landmark['mu'] = [0, 0]
            landmark['sigma'] = np.zeros([2, 2])
            landmark['observed'] = False

            landmarks[i + 1] = landmark

        # add landmarks to particle
        particle['landmarks'] = landmarks

        # add particle to set
        particles.append(particle)

    return particles


def sample_normal_distribution(b):
    samples = np.random.uniform(-b, b, 12)
    return 0.5 * sum(samples)


# def sample_motion_model(odometry, particles):
#     # Updates the particle positions, based on old positions, the odometry
#     # measurements and the motion noise 

#     delta_rot1 = odometry['r1']
#     delta_trans = odometry['t']
#     delta_rot2 = odometry['r2']

#     # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
#     noise = [0.1, 0.1, 0.05, 0.05]

#     new_particles = []
#     for particle in particles:
#         d_rot1_est = delta_rot1 + sample_normal_distribution(noise[0] * abs(delta_rot1) + noise[1] * delta_trans)
#         d_rot2_est = delta_rot2 + sample_normal_distribution(noise[0] * abs(delta_rot2) + noise[1] * delta_trans)
#         d_trans_est = delta_trans + sample_normal_distribution(noise[2] * delta_trans + noise[3] * (abs(delta_rot1) + abs(delta_rot2)))
#         x_new = particle['x'] + d_trans_est * np.cos(particle['theta'] + d_rot1_est)
#         y_new = particle['y'] + d_trans_est * np.sin(particle['theta'] + d_rot1_est)
#         theta_new = particle['theta'] + d_rot1_est + d_rot2_est
#         new_particles.append({'x': x_new, 'y': y_new, 'theta': theta_new})
    
#     return new_particles


def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise 

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    for particle in particles:
        d_rot1_est = delta_rot1 + sample_normal_distribution(noise[0] * abs(delta_rot1) + noise[1] * delta_trans)
        d_rot2_est = delta_rot2 + sample_normal_distribution(noise[0] * abs(delta_rot2) + noise[1] * delta_trans)
        d_trans_est = delta_trans + sample_normal_distribution(noise[2] * delta_trans + noise[3] * (abs(delta_rot1) + abs(delta_rot2)))
        particle['x'] = particle['x'] + d_trans_est * np.cos(particle['theta'] + d_rot1_est)
        particle['y'] = particle['y'] + d_trans_est * np.sin(particle['theta'] + d_rot1_est)
        particle['theta'] = particle['theta'] + d_rot1_est + d_rot2_est


def measurement_model(particle, landmark):
    # Compute the expected measurement for a landmark
    # and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    p_theta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    # calculate expected range measurement
    meas_range_exp = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)
    meas_bearing_exp = math.atan2(ly - py, lx - px) - p_theta

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian h_j of the measurement function h
    # wrt the landmark location

    h_j = np.zeros((2, 2))
    h_j[0, 0] = (lx - px) / h[0]
    h_j[0, 1] = (ly - py) / h[0]
    h_j[1, 0] = (py - ly) / (h[0] ** 2)
    h_j[1, 1] = (lx - px) / (h[0] ** 2)

    return h, h_j


def normalize_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def eval_sensor_model(sensor_data, particles):
    # Correct landmark poses with a measurement and
    # calculate particle weight

    # sensor noise
    q_t = np.array([[0.1, 0],
                    [0, 0.1]])

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    # update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']
        # particle['weight'] = 1.0

        px = particle['x']
        py = particle['y']
        p_theta = particle['theta']

        # loop over observed landmarks
        for i in range(len(ids)):

            # current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]

            # measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            if not landmark['observed']:
                # landmark is observed for the first time

                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                
                z_local = np.array([
                    meas_range * np.cos(meas_bearing),
                    meas_range * np.sin(meas_bearing)
                ])
                R = np.array([
                    [np.cos(p_theta), -np.sin(p_theta)],
                    [np.sin(p_theta),  np.cos(p_theta)]
                ])
                mu = np.array([px, py]) + R @ z_local
                landmark['mu'] = mu

                _, H = measurement_model(particle, landmark)
                landmark['sigma'] =  np.linalg.inv(H) @ q_t @ np.linalg.inv(H).T

                landmark['observed'] = True

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above. 
                # calculate particle weight: particle['weight'] = ...
                
                mu = landmark['mu']
                sigma = landmark['sigma']
                z_hat, H = measurement_model(particle, landmark)

                Q = H @ sigma @ H.T + q_t
                K = sigma @ H.T @ np.linalg.inv(Q)
                
                nu = np.array([
                    meas_range - z_hat[0],
                    normalize_angle(meas_bearing - z_hat[1])
                ])
                mu = mu + K @ nu
                
                I = np.eye(2)
                sigma = (I - K @ H) @ sigma

                landmark['mu'] = mu
                landmark['sigma'] = sigma

                detQ = np.linalg.det(Q)
                norm_const = 1.0 / (2.0 * np.pi * np.sqrt(detQ))
                exponent = -0.5 * (nu.T @ np.linalg.inv(Q) @ nu)
                particle['weight'] *= float(norm_const * np.exp(exponent))

    # normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer


def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle 
    # weights.

    new_particles = []

    weights = [particle['weight'] for particle in particles]
    
    N = len(particles)
    w = [weight / sum(weights) for weight in weights]

    cdf = np.zeros(N)
    acc = 0.0
    for i, weight in enumerate(w):
        acc += weight
        cdf[i] = acc

    r = np.random.uniform(0, 1/N)
    j = 0
    for i in range(N):
        u = r + i / N
        while cdf[j] < u:
            j += 1
        new_particle = copy.deepcopy(particles[j])
        new_particle['weight'] = 1.0 / N
        new_particles.append(new_particle)

    # hint: To copy a particle from particles to the new_particles
    # list, first make a copy:
    # new_particle = copy.deepcopy(particles[i])
    # ...
    # new_particles.append(new_particle)

    return new_particles


def main():

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    num_particles = 100
    num_landmarks = len(landmarks)

    # create particle set
    particles = initialize_particles(num_particles, num_landmarks)

    # run FastSLAM
    for timestep in range(len(sensor_readings) // 2):

        # predict particles by sampling from motion model with odometry info
        sample_motion_model(sensor_readings[timestep, 'odometry'], particles)

        # evaluate sensor model to update landmarks and calculate particle weights
        eval_sensor_model(sensor_readings[timestep, 'sensor'], particles)

        # plot filter state
        plot_state(particles, landmarks)

        # calculate new set of equally weighted particles
        particles = resample_particles(particles)

    plt.show(block=True)


if __name__ == "__main__":
    main()
