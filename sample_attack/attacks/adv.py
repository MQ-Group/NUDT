import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, '..')
import torchattacks

sys.path.insert(0, '..')
import robustbench
from robustbench.data import load_cifar10, load_cifar100
from robustbench.utils import load_model, clean_accuracy

from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from torchattacks import FGSM, PGD, BIM, CW, DeepFool
from .utils import imshow, get_pred


from utils.sse import sse_print


def adv(args):
    device = args.device
    
    try:
        # print(args.data_name)
        if args.data_name == 'cifar10':
            images, labels = load_cifar10(
                                        n_examples=args.selected_samples, # 抽取样本数量
                                        batch_size=args.batch,
                                        num_workers=args.workers,
                                        data_dir=args.data_path)
            dataset = BenchmarkDataset.cifar_10
        elif args.data_name == 'cifar100':
            images, labels = load_cifar100(
                                        n_examples=args.selected_samples, # 抽取样本数量
                                        batch_size=args.batch,
                                        num_workers=args.workers,
                                        data_dir=args.data_path)
            dataset = BenchmarkDataset.cifar_100
        else:
            raise ValueError('数据集不支持, 仅支持数据集: cifar10, cifar100.')

        event = "data_load_validated"
        data = {
            "status": "success",
            "message": "数据集加载完毕.",
            "data name": args.data_name,
            "data path": args.data_path
        }
        sse_print(event, data)
    except Exception as e:
        event = "data_load_validated"
        data = {
            "status": "failure",
            "message": f"{e}",
            "data name": args.data_name,
            "data path": args.data_path
        }
        sse_print(event, data)
        
    # print(images.shape)
    # print(labels.shape)
    
    
    '''
    :param model_name: The name used in the model zoo.
    :param model_dir: The base directory where the models are saved.
    :param dataset: The dataset on which the model is trained.
    :param threat_model: The threat model for which the model is trained.
    :param norm: Deprecated argument that can be used in place of ``threat_model``. If specified, it overrides ``threat_model``
    '''
    model = load_model(
        model_name = args.model_name, # Modas2021PRIMEResNet18 -- cifar100
        model_path = args.model_path,
        dataset = dataset,
        threat_model = ThreatModel.Linf, # 默认值
        norm='Linf'
        ).to(device)
    
    try:
        # print(args.attack_method)
        if args.attack_method == 'fgsm':
            atk = FGSM(model, eps=args.epsilon)
        elif args.attack_method == 'pgd':
            atk = PGD(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, random_start=args.random_start)
        elif args.attack_method == 'bim':
            atk = BIM(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations)
        elif args.attack_method == 'cw':
            atk = CW(model, c=1, kappa=0, steps=args.max_iterations, lr=args.lr)
        elif args.attack_method == 'deepfool':
            atk = DeepFool(model, steps=args.max_iterations, overshoot=0.02)
        else:
            raise ValueError('不支持的攻击方法.')

        event = "adv"
        data = {
            "status": "success",
            "message": "攻击初始化完成.",
            "attack method": args.attack_method
        }
        sse_print(event, data)
    except Exception as e:
        event = "adv"
        data = {
            "status": "failure",
            "message": f"{e}",
            "attack method": args.attack_method
        }
        sse_print(event, data)
        import sys
        sys.exit()
        
    # print(atk)
    
    event = "adversarial_samples_generation_validated"
    for i in range(images.shape[0]):
        data = {
            "status": "success",
            "message": "生成对抗样本...",
            "progress": int(i/args.selected_samples*100),
            "log": f"[{int(i/args.selected_samples*100)}%] 正在生成第{i}张对抗样本, 总共需要生成{args.selected_samples}张."
        }
        sse_print(event, data)

    # adv_images = atk(images, labels)
    
    adv_data_name = f"adv_{args.data_name}.dat"
    adv_data_path = f"{args.output_path}/{adv_data_name}"
    '''
    save_path (str): save_path.
    data_loader (torch.utils.data.DataLoader): data loader.
    verbose (bool): True for displaying detailed information. (Default: True)
    return_verbose (bool): True for returning detailed information. (Default: False)
    save_predictions (bool): True for saving predicted labels (Default: False)
    save_clean_inputs (bool): True for saving clean inputs (Default: False)
    '''
    atk.save(
        data_loader=[(images, labels)], # 在这里面会执行攻击，所以用原始数据
        save_path=adv_data_path,
        verbose=False,
        return_verbose=False,
        save_predictions=False,
        save_clean_inputs=True,
        )
    os.system(f"cp {args.data_yaml} {args.output_path}")
    
    event = "adversarial_samples_generation_validated"
    data = {
        "status": "success",
        "message": "对抗样本生成完成",
        "progress": 100,
        "log": f"[100%] 对抗样本生成完成, 共生成{args.selected_samples}张.",
        "data name": adv_data_name,
        "data path": adv_data_path
    }
    sse_print(event, data)
    
    
    


