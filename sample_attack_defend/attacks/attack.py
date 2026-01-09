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

from torchattacks import FGSM, PGD, BIM, CW, DeepFool, GN, Jitter, Boundary, ZOO, HSJA, NES
from torchdefends import YOPO, TRADES, FREE, FAST
from .utils import imshow, get_pred
from torchvision import transforms

from utils.sse import sse_print

def adv_attack(args):
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

        event = "data_load"
        data = {
            "status": "success",
            "message": "数据集加载完毕.",
            "data name": args.data_name,
            "data path": args.data_path
        }
        sse_print(event, data)
    except Exception as e:
        event = "data_load"
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
        elif args.attack_method == 'gn':
            atk = GN(model, std=args.std)
        elif args.attack_method == 'jitter':
            atk = Jitter(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, scale=args.scale, std=args.std, random_start=args.random_start)
        elif args.attack_method == 'boundary':
            atk = Boundary(model, max_queries=args.max_queries, init_epsilon=args.epsilon, spherical_step=0.01, orthogonal_step=args.step_size, binary_search_steps=args.binary_search_steps)
        elif args.attack_method == 'zoo':
            atk = ZOO(model, max_iterations=args.max_iterations, learning_rate=args.lr, binary_search_steps=args.binary_search_steps, init_const=0.01, beta=0.001, batch_size=128, resolution=1, early_stop_iters=10, abort_early=True)
        elif args.attack_method == 'hsja':
            atk = HSJA(model, max_queries=args.max_queries, norm=args.norm, gamma=0.01, init_num_evals=10, max_num_evals=10, stepsize_search='geometric_progression', num_iterations=args.max_iterations, constraint='L2', batch_size=128)
        elif args.attack_method == 'nes':
            atk = NES(model, max_queries=args.max_queries, epsilon=args.epsilon, learning_rate=args.lr, samples_per_draw=10, sigma=0.001, decay_factor=0.9, norm=args.norm, early_stop=True, loss_func='cross_entropy')
        elif args.attack_method == 'yopo':
            atk = YOPO(model, eps=args.epsilon)
        elif args.attack_method == 'pgdrs':
            atk = PGDRS(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, noise_type=args.noise_type, noise_sd=args.noise_sd, noise_batch_size=5, batch_max=2048)
        elif args.attack_method == 'trades':
            atk = TRADES(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations)
        elif args.attack_method == 'free':
            atk = FREE(model, eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations)
        elif args.attack_method == 'fast':
            atk = FAST(model, eps=args.epsilon)
        else:
            raise ValueError('不支持的攻击方法.')

        event = "adversarial_samples_generation_and_attack_init"
        data = {
            "status": "success",
            "message": "攻击初始化完成.",
            "attack_method": args.attack_method
        }
        sse_print(event, data)
    except Exception as e:
        event = "adversarial_samples_generation_and_attack_init"
        data = {
            "status": "failure",
            "message": f"{e}",
            "attack_method": args.attack_method
        }
        sse_print(event, data)
        import sys
        sys.exit()
        
    # print(atk)
    # adv_images = atk(images, labels)
    
    ori_images_flod = f"{args.output_path}/ori_images"
    adv_images_flod = f"{args.output_path}/adv_images"
    os.makedirs(ori_images_flod, exist_ok=True)
    os.makedirs(adv_images_flod, exist_ok=True)
    
    total_iamges = images.shape[0]
    ori_acc_sum = 0.0
    adv_acc_sum = 0.0
    attack_success_count = 0
    attack_failure_count = 0
    
    for i in range(total_iamges):
        adv_images = atk(images[i:i+1], labels[i:i+1])
        
        ori_acc = clean_accuracy(model, images[i:i+1].to(device), labels[i:i+1].to(device))
        adv_acc = clean_accuracy(model, adv_images.to(device), labels[i:i+1].to(device))
        # print('Acc: %2.2f %%'%(ori_acc*100))
        # print('Acc: %2.2f %%'%(adv_acc*100))
        ori_acc_sum += ori_acc*100
        adv_acc_sum += adv_acc*100
        
        ori_pred_cls = get_pred(model, images[i:i+1], device)
        adv_pred_cls = get_pred(model, adv_images, device)
        # print(ori_pred_cls)
        # print(adv_pred_cls)
        
        if ori_pred_cls != adv_pred_cls:
            # attack_result = '成功'
            attack_result = 'success'
            attack_success_count += 1
        else:
            # attack_result = '失败'
            attack_result = 'failure'
            attack_failure_count += 1
            
        ori_img_save_path = f"{ori_images_flod}/ori_img_{i}_cls_{labels[i].item()}_pred_{ori_pred_cls.item()}.jpg"
        adv_img_save_path = f"{adv_images_flod}/adv_img_{i}_cls_{labels[i].item()}_pred_{adv_pred_cls.item()}.jpg"
        
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(images[i])
        pil_image.save(ori_img_save_path)
        pil_image = to_pil(adv_images[0])
        pil_image.save(adv_img_save_path)
        
        event = "attack_inference"
        data = {
            "message": "正在生成对抗样本并执行攻击...",
            "progress": int(i/total_iamges*100),
            # "log": f"[{int(i/total_iamges*100)}%] 正在生成对抗样本并执行攻击... 原始样本: {ori_img_save_path}, 原始样本预测准确率: {ori_acc*100:.2f}%, 原始样本预测类别: {ori_pred_cls.item()}, 对抗样本: {adv_img_save_path}, 对抗样本预测准确率: {adv_acc*100:.2f}%, 对抗样本预测类别: {ori_pred_cls.item()}, 原始样本实际类别: {labels[i].item()}, 攻击结果: {attack_result}"
            "log": f"[{int(i/total_iamges*100)}%] 正在生成对抗样本并执行攻击...",
            "details": {
                "original_sample": {
                    "accuracy": f"{ori_acc*100:.2f}%",
                    "predict_class": ori_pred_cls.item(),
                    "file_path": ori_img_save_path
                },
                "adversarial_sample": {
                    "accuracy": f"{adv_acc*100:.2f}%",
                    "predict_class": adv_pred_cls.item(),
                    "file_path": adv_img_save_path
                },
                "actual_class": labels[i].item(),
                "attack_result": attack_result
            }
        }
        sse_print(event, data)
    
    event = "final_result"
    data = {
        "message": "生成对抗样本并执行攻击完成.",
        "progress": 100,
        "log": f"[100%] 生成对抗样本并执行攻击完成.",
        "details": {
            "attack_method": args.attack_method,
            "original_samples": ori_images_flod,
            "adversarial_samples": adv_images_flod,
            "summary": {
                "total_iamges": total_iamges,
                "task_success_count": attack_success_count,
                "task_failure_count": attack_failure_count,
                "original_samples_accuracy": f"{ori_acc_sum/total_iamges:.2f}%",
                "adversarial_samples_accuracy": f"{adv_acc_sum/total_iamges:.2f}%"
            }
        }
    }
    sse_print(event, data)
    
