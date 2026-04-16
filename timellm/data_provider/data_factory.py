"""
EEG 数据加载工厂
支持 DEAP 和 SEED 两个数据集，并提供情绪相关的通道选择策略。
"""

from torch.utils.data import DataLoader

from data_provider.data_loader_eeg import Dataset_DEAP, Dataset_SEED, ChannelSelector

data_dict = {
    'DEAP': Dataset_DEAP,
    'SEED': Dataset_SEED,
}


def data_provider(args, flag):
    if args.data not in data_dict:
        raise ValueError(
            f"不支持的数据集: {args.data}，可选: {list(data_dict.keys())}")

    Data = data_dict[args.data]

    # EEG 数据集参数
    if flag == 'test':
        shuffle_flag = False
        drop_last = False  # 测试集不丢弃最后一批
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    if args.data == 'DEAP':
        deap_args = {
            'root_path': args.root_path,
            'flag': flag,
            'seq_len': args.seq_len,
            'pred_len': args.pred_len if hasattr(args, 'pred_len') else 0,
            'label_len': args.label_len if hasattr(args, 'label_len') else 0,
            'n_class': args.num_class,
            'classification_type': getattr(args, 'classification_type', 'valence'),
            'subject_list': getattr(args, 'subject_list', None),
            'overlap': getattr(args, 'overlap', 0),
            'normalize': getattr(args, 'normalize', True),
            'filter_freq': getattr(args, 'filter_freq', None),
            'sampling_rate': getattr(args, 'sampling_rate', 128),
            'channel_selection': getattr(args, 'channel_selection', 'auto'),
            'use_channel_selection': getattr(args, 'use_channel_selection', True)
        }
        data_set = Data(**deap_args)

        # 根据实际通道选择结果更新模型输入通道数
        if hasattr(data_set, 'n_channels'):
            args.enc_in = data_set.n_channels
            args.dec_in = data_set.n_channels
            args.c_out = data_set.n_channels
            print(f"已更新模型输入通道数为: {data_set.n_channels}")

    else:  # SEED
        seed_args = {
            'root_path': args.root_path,
            'flag': flag,
            'seq_len': args.seq_len,
            'pred_len': args.pred_len if hasattr(args, 'pred_len') else 0,
            'label_len': args.label_len if hasattr(args, 'label_len') else 0,
            'n_class': args.num_class,
            'subject_list': getattr(args, 'subject_list', None),
            'overlap': getattr(args, 'overlap', 0),
            'normalize': getattr(args, 'normalize', True),
            'filter_freq': getattr(args, 'filter_freq', None),
            'sampling_rate': getattr(args, 'sampling_rate', 200)
        }
        data_set = Data(**seed_args)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader


def print_available_channel_groups():
    """打印所有可用的EEG通道组"""
    print("\n" + "="*80)
    print("DEAP数据集可用的情绪相关通道组:")
    print("="*80)
    ChannelSelector.print_available_groups()
    print("="*80)


def get_recommended_channel_selection(classification_type, n_class):
    """
    根据分类任务推荐通道选择策略
    
    参数:
        classification_type: 'valence', 'arousal', 或其他
        n_class: 分类数量
        
    返回:
        推荐的通道选择策略名称
    """
    recommendations = {
        ('valence', 2): 'valence_specific',
        ('arousal', 2): 'arousal_specific', 
        ('valence', 4): 'frontal_emotion',
        ('arousal', 4): 'comprehensive_emotion',
        ('both', 4): 'comprehensive_emotion'
    }
    
    key = (classification_type, n_class)
    recommended = recommendations.get(key, 'comprehensive_emotion')
    
    print(f"\n推荐的通道选择策略:")
    print(f"  分类类型: {classification_type}")
    print(f"  分类数量: {n_class}")
    print(f"  推荐策略: {recommended}")
    
    return recommended


# 测试代码
if __name__ == "__main__":
    # 创建一个简单的参数对象来测试
    class Args:
        def __init__(self):
            # 基础参数
            self.data = 'DEAP'  # 或 'SEED'
            self.root_path = r"D:\文件\文件\HKU\Dissertation\dataset\DEAP\data_preprocessed_python"
            self.seq_len = 256
            self.batch_size = 32
            self.num_workers = 0
            self.embed = 'timeF'

            # EEG特定参数
            self.num_class = 2
            self.classification_type = 'valence'
            self.overlap = 128
            self.normalize = True
            self.filter_freq = (0.5, 45)
            self.sampling_rate = 128
            self.subject_list = ['s01', 's02']  # 测试用少量被试

            # 通道选择参数
            self.channel_selection = 'auto'  # 自动选择
            self.use_channel_selection = True

            # 其他可能需要的参数
            self.pred_len = 0
            self.label_len = 0
            self.enc_in = 32  # 初始值，会被自动更新
            self.dec_in = 32
            self.c_out = 32

    # 打印所有可用的通道组
    print_available_channel_groups()
    
    # 获取推荐的通道选择策略
    recommended = get_recommended_channel_selection('valence', 2)

    # 测试DEAP数据集
    print("\n" + "="*60)
    print("测试DEAP数据集的data_provider...")
    print("="*60)
    args = Args()
    args.data = 'DEAP'
    args.channel_selection = recommended

    train_data, train_loader = data_provider(args, 'train')
    print(f"训练集大小: {len(train_data)}")
    print(f"批次数量: {len(train_loader)}")
    print(f"更新后的模型输入通道数: {args.enc_in}")

    # 获取一个批次
    for batch in train_loader:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        print(f"\n批次形状:")
        print(f"  - batch_x: {batch_x.shape}")
        print(f"  - batch_y: {batch_y.shape}")
        print(f"  - batch_x_mark: {batch_x_mark.shape}")
        print(f"  - batch_y_mark: {batch_y_mark.shape}")
        break

    # 测试不同的通道选择策略
    print("\n" + "="*60)
    print("测试不同通道选择策略的效果...")
    print("="*60)
    
    strategies = ['frontal_emotion', 'frontal_asymmetry', 'valence_specific', 'comprehensive_emotion']
    
    for strategy in strategies:
        print(f"\n使用策略: {strategy}")
        args.channel_selection = strategy
        
        try:
            train_data, train_loader = data_provider(args, 'train')
            print(f"  - 数据集大小: {len(train_data)}")
            print(f"  - 通道数: {args.enc_in}")
            
            # 获取通道信息
            if hasattr(train_data, 'get_channel_info'):
                channel_info = train_data.get_channel_info()
                print(f"  - 使用的通道: {', '.join(channel_info['channel_names'])}")
                
        except Exception as e:
            print(f"  - 错误: {e}")

    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)