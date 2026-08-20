import ml_collections
from vmoe.configs.vmoe_paper import common

NUM_CLASSES = 6
BATCH_SIZE = 16
PP_COMMON = f'value_range(-1,1)|onehot({NUM_CLASSES}, inkey="label", outkey="labels")|keep("image", "labels")'


def get_config():
  """Config to train V-MoE model on Potato Leaf Disease dataset with RAM Preloading & Exact Augmentation."""
  config = common.get_base_config()

  config.dataset = ml_collections.ConfigDict()
  
  # Cấu hình tập Train (Áp dụng Data Augmentation & Normalization)
  config.dataset.train = ml_collections.ConfigDict({
      'name': 'potato_csv',
      'split': 'train',
      'batch_size': BATCH_SIZE,
      'process': f'flip_lr|{PP_COMMON}',
      'csv_path': '',
      'data_dir': '',
      'use_cache': True,
      'shuffle_buffer': 1000,
  })

  # Cấu hình tập Validation
  config.dataset.val = ml_collections.ConfigDict({
      'name': 'potato_csv',
      'split': 'val',
      'batch_size': BATCH_SIZE,
      'process': f'{PP_COMMON}',
      'csv_path': '',
      'data_dir': '',
      'use_cache': True,
      'cache': 'batched',
  })

  # Cấu hình tập Test
  config.dataset.test = ml_collections.ConfigDict({
      'name': 'potato_csv',
      'split': 'test',
      'batch_size': BATCH_SIZE,
      'process': f'{PP_COMMON}',
      'csv_path': '',
      'data_dir': '',
      'use_cache': True,
      'cache': 'batched',
  })

  # Loss & Mô hình VMoE (8 layers, 16 experts)
  config.loss = ml_collections.ConfigDict()
  config.loss.name = 'softmax_xent'
  
  config.description = 'ViT-S/16, E=16, K=2, Last 2'
  config.model = get_vmoe_params()
  config.optimizer = get_optimizer_params()
  config.train_epochs = 100
  config.patience = 20

  # Partitioning cho JAX Mesh
  config.num_expert_partitions = config.model.encoder.moe.num_experts
  config.params_axis_resources = [('Moe/Mlp/.*', ('expert',))]
  config.extra_rng_keys = ('dropout', 'gating')

  return config


def get_vmoe_params():
  config = ml_collections.ConfigDict()
  config.name = 'VisionTransformerMoe'
  config.num_classes = NUM_CLASSES
  config.patch_size = (16, 16)
  config.hidden_size = 384
  config.classifier = 'token'
  config.representation_size = None
  
  # Cấu hình Encoder
  config.encoder = ml_collections.ConfigDict()
  config.encoder.num_layers = 8
  config.encoder.num_heads = 6
  config.encoder.mlp_dim = 1536
  config.encoder.dropout_rate = 0.1
  config.encoder.attention_dropout_rate = 0.0

  # Cấu hình MoE với 16 Experts
  config.encoder.moe = ml_collections.ConfigDict()
  config.encoder.moe.layers = (6, 7)
  config.encoder.moe.num_experts = 16
  config.encoder.moe.group_size = 1
  config.encoder.moe.router = ml_collections.ConfigDict({
      'name': 'NoisyTopExpertsPerItemRouter',
      'num_selected_experts': 2,
      'importance_loss_weight': 0.005,
      'load_loss_weight': 0.005,
      'dispatcher': {
          'name': 'einsum',
          'capacity_factor': 1.05,
      }
  })
  return config


def get_optimizer_params():
  config = ml_collections.ConfigDict()
  config.name = 'adam'
  config.learning_rate = ml_collections.ConfigDict({
      'schedule': 'warmup_cosine_decay',
      'peak_value': 5e-6,
      'warmup_steps': 500,
      'end_value': 1e-7,
  })
  config.weight_decay = 0.01
  return config
