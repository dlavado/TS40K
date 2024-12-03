import torch
import wandb
import sys
import pytorch_lightning as pl

# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
import utils.my_utils as su
import utils.constants as consts
from scripts import main as m
import utils.pointcloud_processing as eda


#####################################################################
# PARSER
#####################################################################

def replace_variables(string):
    """
    Replace variables marked with '$' in a string with their corresponding values from the local scope.

    Args:
    - string: Input string containing variables marked with '$'

    Returns:
    - Updated string with replaced variables
    """
    import re

    pattern = r'\${(\w+)}'
    matches = re.finditer(pattern, string)

    for match in matches:
        variable = match.group(1)
        value = locals().get(variable)
        if value is None:
            value = globals().get(variable)

        if value is not None:
            string = string.replace(match.group(), str(value))
        else:
            raise ValueError(f"Variable '{variable}' not found.")

    return string



def predict(model:pl.LightningModule, data_module:pl.LightningDataModule):

    test_loader = data_module.test_dataloader()

    metrics = su.init_metrics(
        num_classes=wandb.config.num_classes,
        ignore_index=wandb.config.ignore_index,
    ).to(consts.device)

    for i, batch in enumerate(test_loader):
        # batch to device
        if i < 318:
            continue # skip the first 42 batches
        batch = tuple([s.to(consts.device) for s in batch])
    
        loss, pred, y = model.evaluate(batch, stage='test', metric=metrics)

        print(pred.shape)
        print(pred.unique())

        xyz = batch[-1] # pt_locs

        print(f"batch {i}; sample 0")

        print(f"Cross Entropy loss: {loss.item()}")

        if loss.item() > 0.5:
            continue

        # print metrics:
        print(f"{'='*50} METRICS {'='*50}")
        for key, value in metrics.items():
            print(f"{key}: {value.compute()}")

        metrics.reset()

        # if loss.item() > 0.5 or loss.item() < 0.2: # visualize the interesting cases

        if xyz.ndim == 3: # batched
            xyz = xyz[0]
            y = y[0]
            pred = pred[0]
    
        xyz = xyz.squeeze().cpu().numpy()
        pynt = eda.np_to_ply(xyz)

        ### ground truth
        y = y.reshape(-1).cpu().numpy()
        eda.color_pointcloud(pynt, y, use_preset_colors=True)
        eda.visualize_ply([pynt], window_name="Ground Truth")

        ### prediction
        pred = pred.reshape(-1).cpu().numpy()
        eda.color_pointcloud(pynt, pred, use_preset_colors=True)
        eda.visualize_ply([pynt], window_name="Prediction")
    




def main():
    # ------------------------
    # 0 INIT CKPT PATH
    # ------------------------

    if not wandb.config.resume_from_checkpoint:
        ckpt_dir = os.path.join(wandb.run.dir, "checkpoints") 
    else:
        ckpt_dir = wandb.config.checkpoint_dir

    ckpt_path = os.path.join(ckpt_dir, wandb.config.resume_checkpoint_name + '.ckpt')

    ckpt_dir = replace_variables(ckpt_dir)
    ckpt_path = replace_variables(ckpt_path)

    # ------------------------
    # 1 INIT BASE CRITERION
    # ------------------------
    class_densities = torch.tensor([0.0702, 0.3287, 0.4226, 0.1495, 0.0046, 0.0244], dtype=torch.float32)
    class_weights = 1 / class_densities
    class_weights[0] = 0.0 # ignore noise class
    class_weights = class_weights / class_weights.mean()
    criterion = m.init_criterion(class_weights=class_weights)

    # ------------------------
    # 2 INIT MODEL
    # ------------------------
    model = m.init_model(wandb.config.model, criterion)
    if wandb.config.get('geneo_criterion', False):
        m.init_GENEO_loss(model, base_criterion=criterion)
    model = m.resume_from_checkpoint(ckpt_path, model, class_weights)
    model = model.to(consts.device)
    model.eval()

    # ------------------------
    # 3 INIT DATA MODULE
    # ------------------------

    dataset_name = wandb.config.dataset       
    if dataset_name == 'ts40k':
        data_path = consts.TS40K_FULL_PATH
        if wandb.config.preprocessed:
            data_path = consts.TS40K_FULL_PREPROCESSED_PATH
            if idis_mode:
                data_path = consts.TS40K_FULL_PREPROCESSED_IDIS_PATH
            elif smote_mode:
                data_path = consts.TS40K_FULL_PREPROCESSED_SMOTE_PATH
        data_module = m.init_ts40k(data_path, wandb.config.preprocessed)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    wandb.config.update({'data_path': data_path}, allow_val_change=True) # override data path

    print(f"\n=== Data Module {dataset_name.upper()} initialized. ===\n")
    print(f"{data_module}")
    print(data_path)
    

    ####### if Pytorch Lightning Trainer is not called, setup() needs to be called manually
    data_module.setup('test')
    

    # ------------------------
    # 4 PREDICT
    # ------------------------
    predict(model, data_module)

if __name__ == "__main__":
    # --------------------------------
    import warnings
    from datetime import datetime
    import os

    su.fix_randomness()
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('medium')


    # is cuda available
    print(f"{'='*50} CUDA available: {torch.cuda.is_available()} {'='*50}")
    # get device specs
    print(f"{'='*3}> Device specs: {torch.cuda.get_device_properties(0)}")

    main_parser = su.main_arg_parser().parse_args()

    model_name = main_parser.model
    dataset_name = main_parser.dataset
    project_name = f"TS40K_SoA"

    idis_mode = main_parser.idis
    smote_mode = main_parser.smote

    # config_path = get_experiment_config_path(model_name, dataset_name)
    experiment_path = consts.get_experiment_dir(model_name, dataset_name)

    os.environ["WANDB_DIR"] = os.path.abspath(os.path.join(experiment_path, 'wandb'))
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # idk man

    # default mode
    sweep_config = os.path.join(experiment_path, 'defaults_config.yml')

    print("wandb init.")

    wandb.init(project=project_name, 
            dir = experiment_path,
            name = f'{project_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            config=sweep_config,
            mode='disabled',
    ) 

    main()       
