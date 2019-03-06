from opmimization.particle_swarm import ParticleSwarmOptimizer



#function that models the problem
def fitness_function(position):
    if position.shape[0] == 2:
        return position[0]**2 + position[1]**2 + 1
    if position.shape[0] == 3:
        return position[0]**2 + position[1]**2 + position[2]**2 + 1


pso = ParticleSwarmOptimizer(fitness_function, 3, 1, 0.000001, 30)

pso.optimize(50, print_opt=True)