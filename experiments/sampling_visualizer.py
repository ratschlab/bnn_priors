import numpy as np
import torch
import scipy.stats
import sys
import pygame

from bnn_priors.models import DenseNet, AbstractModel
from bnn_priors.inference import SGLDRunner
from bnn_priors import prior
from bnn_priors.mcmc import SGLD, VerletSGLD, HMC


temperature = 1
model = prior.Normal(torch.Size([1]), 0., 1.)
def potential():
    model.zero_grad()
    loss = -model.log_prob()
    loss.backward()
    return loss

if not len(sys.argv) == 2:
    print(f"Usage: {sys.argv[0]} (HMC|SGLD|VerletSGLD)")
    sys.exit(1)

if sys.argv[1].lower() == "sgld":
    sgld = SGLD(model.parameters(), lr=2/4, num_data=1,
                momentum=3/4, temperature=temperature)
elif sys.argv[1].lower() == "verletsgld":
    sgld = VerletSGLD(model.parameters(), lr=2/4, num_data=1,
                      momentum=3/4, temperature=temperature)
elif sys.argv[1].lower() == "hmc":
    sgld = HMC(model.parameters(), lr=1/2, num_data=1)
else:
    raise ValueError(sys.argv[1])

model.sample()
sgld.sample_momentum()

pygame.init()
sz = 720
radius = sz/20
screen = pygame.display.set_mode((sz//3*4, sz))
center_x, center_y = sz//3*2- sz//3, sz//2
center_x2, center_y2 = sz//3*2 + sz//3, sz//2

pygame.display.set_caption("hello")

first = True
x = y = None

screen.fill((0xcc, 0xcc, 0xcc))

running = True
step = 0
while running:
    if isinstance(sgld, HMC):
        if step % 4 == 0:
            sgld.sample_momentum()
            sgld.initial_step(potential)
            sgld.final_step(potential)
        else:
            sgld.initial_step(potential)
            sgld.final_step(potential)
    elif isinstance(sgld, VerletSGLD):
        sgld.initial_step(potential)
        sgld.final_step(potential)
    else:
        sgld.step(potential)
    prev_x = x
    prev_y = y
    for p, state in sgld.state.items():
        x = p.item()
        y = state['momentum_buffer'].item()

    if first:
        prev_x = x
        prev_y = y

    start_pos = tuple(map(int, (prev_x*radius+center_x, prev_y*radius+center_y)))
    end_pos = tuple(map(int, (x*radius+center_x, y*radius+center_y)))

    pygame.draw.aaline(screen, (0, 0, 0), start_pos, end_pos)
    pygame.draw.circle(screen, (0, 0, 0), end_pos, 3)

    x, y =torch.randn(2)
    end_pos = tuple(map(int, (x*radius+center_x2, y*radius+center_y2)))
    pygame.draw.circle(screen, (255, 0, 0), end_pos, 3)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            else:
                sgld.sample_momentum()

    pygame.display.flip()
    step += 1

pygame.quit()
