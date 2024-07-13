import pygame
import random
import numpy as np
import os

pygame.init()

WIDTH = 1200
HEIGHT = 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino AI")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Dino:
    def __init__(self, color):
        self.x = 50
        self.y = HEIGHT - 60
        self.vel_y = 0
        self.width = 40
        self.height = 60
        self.is_jumping = False
        self.is_ducking = False
        self.color = color
        self.score = 0
        self.alive = True
        self.jumps = 0
        self.ducks = 0

    def jump(self):
        if not self.is_jumping and not self.is_ducking:
            self.vel_y = -15
            self.is_jumping = True
            self.jumps += 1

    def duck(self):
        if not self.is_jumping:
            self.is_ducking = True
            self.height = 30
            self.ducks += 1

    def stand(self):
        self.is_ducking = False
        self.height = 60

    def update(self):
        self.vel_y += 1
        self.y += self.vel_y
        if self.y > HEIGHT - 60:
            self.y = HEIGHT - 60
            self.is_jumping = False
            self.vel_y = 0
        self.score += 1

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))

class Obstacle:
    def __init__(self, x):
        self.x = x
        self.width = 20
        self.type = random.choice(["cactus", "bird"])
        if self.type == "cactus":
            self.height = random.randint(30, 50)
            self.y = HEIGHT - 60
        else:
            self.height = 20
            self.y = HEIGHT - 60 - random.randint(20, 80)

    def update(self, speed):
        self.x -= speed

    def draw(self, screen):
        color = BLACK if self.type == "cactus" else (0, 0, 255)
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))

class Brain:
    def __init__(self, input_size=22, hidden_size=32, output_size=3):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, hidden_size)
        self.weights3 = np.random.randn(hidden_size, output_size)

    def predict(self, inputs):
        layer1 = np.maximum(0, np.dot(inputs, self.weights1))
        layer2 = np.maximum(0, np.dot(layer1, self.weights2))
        return 1 / (1 + np.exp(-np.dot(layer2, self.weights3)))

def get_inputs(dino, obstacles):
    inputs = [
        dino.y / HEIGHT,
        dino.vel_y / 10,
        int(dino.is_jumping),
        int(dino.is_ducking)
    ]
    for i in range(3):
        if i < len(obstacles):
            obstacle = obstacles[i]
            inputs.extend([
                obstacle.x / WIDTH,
                obstacle.y / HEIGHT,
                obstacle.width / WIDTH,
                obstacle.height / HEIGHT,
                1 if obstacle.type == "bird" else 0,
                (obstacle.x - dino.x) / WIDTH
            ])
        else:
            inputs.extend([1, 0, 0, 0, 0, 1])
    return np.array(inputs)

def game(dinos, brains):
    clock = pygame.time.Clock()
    obstacles = []
    max_score = 0
    speed = 10
    frames = 0

    while any(dino.alive for dino in dinos):
        frames += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return [dino.score for dino in dinos]

        if len(obstacles) == 0 or obstacles[-1].x < WIDTH - random.randint(200, 500):
            obstacles.append(Obstacle(WIDTH))

        obstacles = [obstacle for obstacle in obstacles if obstacle.x > -20]

        screen.fill(WHITE)

        for dino, brain in zip(dinos, brains):
            if dino.alive:
                inputs = get_inputs(dino, obstacles)
                output = brain.predict(inputs)
                action = np.argmax(output)
                if action == 0:
                    dino.jump()
                elif action == 1:
                    dino.duck()
                else:
                    dino.stand()

                dino.update()
                dino.draw(screen)

                for obstacle in obstacles:
                    if (dino.x < obstacle.x + obstacle.width and 
                        dino.x + dino.width > obstacle.x and 
                        dino.y + dino.height > obstacle.y and
                        dino.y < obstacle.y + obstacle.height):
                        dino.alive = False

                max_score = max(max_score, dino.score)

        for obstacle in obstacles:
            obstacle.update(speed)
            obstacle.draw(screen)

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Max Score: {max_score}", True, BLACK)
        speed_text = font.render(f"Speed: {speed:.1f}", True, BLACK)
        screen.blit(score_text, (10, 10))
        screen.blit(speed_text, (10, 50))

        pygame.display.flip()
        clock.tick(60)

        if frames % 1000 == 0:
            speed += 0.5

    return [dino.score for dino in dinos]

def calculate_novelty(dino, population):
    behavior = np.array([dino.score, dino.jumps, dino.ducks])
    distances = [np.linalg.norm(behavior - np.array([d.score, d.jumps, d.ducks])) for d in population]
    return np.mean(sorted(distances)[:15])

def save_best_brain(brain, generation):
    filename = 'best_brain.npz'
    if os.path.exists(filename):
        os.remove(filename)
    np.savez(filename, 
             weights1=brain.weights1, 
             weights2=brain.weights2,
             weights3=brain.weights3,
             generation=generation)
    print(f"Mejor cerebro guardado en la generación {generation}")

def load_best_brain():
    filename = 'best_brain.npz'
    if os.path.exists(filename):
        data = np.load(filename)
        brain = Brain(input_size=22, hidden_size=32, output_size=3)
        brain.weights1 = data['weights1']
        brain.weights2 = data['weights2']
        brain.weights3 = data['weights3']
        generation = int(data['generation'])
        print(f"Cerebro cargado de la generación {generation}")
        return brain, generation
    return None, 0

def genetic_algorithm(population_size=100, generations=1000):
    best_brain, start_generation = load_best_brain()
    if best_brain is None:
        population = [Brain(input_size=22, hidden_size=32, output_size=3) for _ in range(population_size)]
        best_score = 0
    else:
        population = [best_brain] + [Brain(input_size=22, hidden_size=32, output_size=3) for _ in range(population_size - 1)]
        best_score = 0  # Reiniciamos el best_score ya que estamos comenzando una nueva ejecución

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(population_size)]
    
    for generation in range(start_generation, generations):
        dinos = [Dino(color) for color in colors]
        scores = game(dinos, population)
        
        novelty_scores = [calculate_novelty(dino, dinos) for dino in dinos]
        
        combined_scores = [score + 0.5 * novelty for score, novelty in zip(scores, novelty_scores)]
        
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        print(f"Generation {generation + 1}: Max Score = {max_score}, Avg Score = {avg_score}")
        
        if max_score > best_score:
            best_score = max_score
            best_brain = population[scores.index(max_score)]
            save_best_brain(best_brain, generation + 1)
        
        sorted_population = [x for _, x in sorted(zip(combined_scores, population), key=lambda pair: pair[0], reverse=True)]
        
        new_population = sorted_population[:10]  # Elitism: keep top 10
        
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(sorted_population[:population_size // 2], 2)
            child = Brain(input_size=22, hidden_size=32, output_size=3)
            
            # Uniform crossover
            for attr in ['weights1', 'weights2', 'weights3']:
                mask = np.random.rand(*getattr(child, attr).shape) < 0.5
                setattr(child, attr, np.where(mask, getattr(parent1, attr), getattr(parent2, attr)))
            
            # Mutation
            mutation_rate = 0.1
            mutation_strength = 0.2
            for attr in ['weights1', 'weights2', 'weights3']:
                if random.random() < mutation_rate:
                    mutation = np.random.randn(*getattr(child, attr).shape) * mutation_strength
                    setattr(child, attr, getattr(child, attr) + mutation)
            
            new_population.append(child)
        
        population = new_population

    return best_brain, best_score

def draw_button(screen, text, x, y, width, height, color, text_color):
    pygame.draw.rect(screen, color, (x, y, width, height))
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=(x + width/2, y + height/2))
    screen.blit(text_surface, text_rect)

def main_menu():
    clock = pygame.time.Clock()
    running = True
    training = False
    best_brain = None
    best_score = 0

    while running:
        screen.fill(WHITE)
        
        draw_button(screen, "Play/Train", 450, 100, 300, 50, GREEN, BLACK)
        draw_button(screen, "Reset", 450, 170, 300, 50, RED, BLACK)
        #draw_button(screen, "Delete Brain", 450, 240, 300, 50, RED, BLACK)
        #draw_button(screen, "Quit", 450, 310, 300, 50, BLUE, BLACK)
        draw_button(screen, "Quit", 450, 240, 300, 50, BLUE, BLACK)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if 450 <= mouse_pos[0] <= 750:
                    if 100 <= mouse_pos[1] <= 150:  # Play/Train
                        if not training:
                            training = True
                            best_brain, best_score = genetic_algorithm()
                            training = False
                    elif 170 <= mouse_pos[1] <= 220:  # Reset
                        best_brain = None
                        best_score = 0
                        if os.path.exists('best_brain.npz'):
                            os.remove('best_brain.npz')
                        print("Reset completo. Cerebro eliminado.")
                    elif 240 <= mouse_pos[1] <= 290:  # Delete Brain
                        if os.path.exists('best_brain.npz'):
                            os.remove('best_brain.npz')
                            print("Cerebro eliminado.")
                        else:
                            print("No hay cerebro para eliminar.")
                    elif 310 <= mouse_pos[1] <= 360:  # Quit
                        running = False

        if training:
            font = pygame.font.Font(None, 48)
            text = font.render("Training in progress...", True, BLACK)
            text_rect = text.get_rect(center=(WIDTH/2, HEIGHT/2))
            screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main_menu()