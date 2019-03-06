# particle swarm optimizer
# based on the article:
# https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6
import random
import numpy as np 


class Particle():
    """
    A particle of the swarm
    """
    def __init__(self, dimensions):
        """
        Init
            :param self: 
            :param dimensions: number of dimensions for the particle
        """
        self._initialize_position(dimensions)
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.zeros(dimensions)

    def _initialize_position(self, dimensions):
        """
        Initialize to a random position
            :param self: 
            :param dimensions: number of dimensions to optimize
        """
        self.position = np.zeros(dimensions)
        for d in range(dimensions):
            self.position[d] = (-1)**(bool(random.getrandbits(1))) * \
                                random.random()*50

    def __str__(self):
        print("I am at ", self.position, " meu pbest is ", self.pbest_position)
    
    def move(self):
        """
        Update the particle position
            :param self: 
        """
        self.position = self.position + self.velocity


class Space():
    """
    Space where the particles displace
    """
    def __init__(self, function, dimensions, target, target_error, n_particles,
                 inertia=0.5, acc_coefficient_1=0.8, acc_coefficient_2=0.9):
        """
        Init
            :param self: 
            :param function: function to explore
            :param dimensions: number of dimensions for the space
            :param target: target value
            :param target_error: acceptable error
            :param n_particles: number of particles
            :param inertia=0.5: inertial parameter
            :param acc_coefficient_1=0.8: coefficient personal best value
            :param acc_coefficient_2=0.9: coefficient social best value
        """
        self.function = function
        self.dimensions = dimensions
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.inertia = inertia
        self.acc_coefficient_1 = acc_coefficient_1
        self.acc_coefficient_2 = acc_coefficient_2
        self.gbest_value = float('inf')
        self._initialize_gbest_position()
        self._generate_particles()

    def _initialize_gbest_position(self):
        self.gbest_position = np.zeros(self.dimensions)
        for d in range(self.dimensions):
            self.gbest_position[d] = random.random()*50

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()

    def _generate_particles(self):
        self.particles = \
            [Particle(self.dimensions) for _ in range(self.n_particles)]
   
    def _fitness(self, particle):
        return self.function(particle.position)

    def set_pbest(self):
        for particle in self.particles:
            fitness_cadidate = self._fitness(particle)
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position
            
    def set_gbest(self):
        for particle in self.particles:
            best_fitness_cadidate = self._fitness(particle)
            if(self.gbest_value > best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position

    def move_particles(self):
        for particle in self.particles:
            new_velocity = (self.inertia*particle.velocity) + \
                (self.acc_coefficient_1*random.random()) * \
                (particle.pbest_position - particle.position) + \
                (random.random()*self.acc_coefficient_2) * \
                (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()


class ParticleSwarmOptimizer():
    """
    A Particle Swarm Optimizer.
    """
    def __init__(self, function, dimensions, target, target_error, n_particles,
                 inertia=0.5, acc_coefficient_1=0.8, acc_coefficient_2=0.9):
        """
        Init
            :param self: 
            :param function: function to explore
            :param dimensions: number of dimensions to optimize
            :param target: target value
            :param target_error: acceptable error
            :param n_particles: number of particles
            :param inertia=0.5: inertial parameter
            :param acc_coefficient_1=0.8: coefficient personal best value
            :param acc_coefficient_2=0.9: coefficient social best value
        """
        self.search_space = Space(function, dimensions, target,
                                  target_error, n_particles)
        self.inertia = inertia
        self.acc_coefficient_1 = acc_coefficient_1
        self.acc_coefficient_2 = acc_coefficient_2

    def optimize(self, n_iterations, print_opt=False):
        """
        Run the optimizer
            :param self: 
            :param n_iterations: number of iterations
            :param print_opt=False: print the result
        """
        iteration = 0
        while(iteration < n_iterations):
            self.search_space.set_pbest()    
            self.search_space.set_gbest()

            if(abs(self.search_space.gbest_value - \
                  self.search_space.target) <= self.search_space.target_error):
                # finish the search
                break

            self.search_space.move_particles()
            iteration += 1
        if print_opt:
            print("The best solution is: ", self.search_space.gbest_position,
                " in n_iterations: ", iteration)

 