import torch

device = torch.device(f'cuda:{int(0)}' if torch.cuda.is_available() else 'cpu')
image=torch.load('/Users/liuji/PycharmProjects/CSRL/environments_and_constraints/lunar_lander/trained_agent_files'
                 '/p0_train.pt', map_location=device)

print(image)