# from src.configs.disha.file import file_config as disha_file_config
# from src.configs.lora.file import file_config as lora_file_config
# from src.configs.pissa.file import file_config as pissa_file_config
from src.configs.prepare.file import file_config as prepare_file_config
# from src.configs.pretrain.file import file_config as pretrain_file_config
# from src.configs.state_tuning.file import file_config as state_tuning_file_config

# from src.configs.disha.model import model_config as disha_model_config
# from src.configs.lora.model import model_config as lora_model_config
# from src.configs.pissa.model import model_config as pissa_model_config
from src.configs.prepare.model import model_config as prepare_model_config
# from src.configs.pretrain.model import model_config as pretrain_model_config
# from src.configs.state_tuning.model import model_config as state_tuning_model_config

# from src.configs.disha.train import train_config as disha_train_config
# from src.configs.lora.train import train_config as lora_train_config
# from src.configs.pissa.train import train_config as pissa_train_config
# from src.configs.prepare.train import train_config as prepare_train_config
from src.configs.pretrain.train import train_config as pretrain_train_config
# from src.configs.state_tuning.train import train_config as state_tuning_train_config

file_config = prepare_file_config
model_config = prepare_model_config
train_config = pretrain_train_config
