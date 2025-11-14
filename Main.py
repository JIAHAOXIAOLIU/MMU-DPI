from model import execute_workflow
import torch

if __name__ == '__main__':
    execute_workflow('BindingDB_Kd', phase="train", batch_size=32, epochs=25, learning_rate=5e-4, lr_step_size=10, seed_id=10, device=torch.device('cpu'), mixup=False)
    # execute_workflow('DAVIS')
    # execute_workflow('BindingDB_Kd')
    # execute_workflow('KIBA')
